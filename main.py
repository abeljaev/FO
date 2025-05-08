import os
import glob
import json
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz  # Для автоматической загрузки моделей
from loguru import logger
from PIL import Image
import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
)  # Останутся для кастомной обертки, если Zoo не сработает

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ (Минимальные) ---
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
# logger.add("processing_critical.log", level="ERROR") # Опционально: только ошибки в файл

# --- ИМПОРТ КОНФИГУРАЦИИ ---
try:
    from config import (
        PATH_TO_SPLIT,
        PATH_TO_PREDICTIONS,
        SIZE_REQUIREMENTS,
        OUR_TO_MODEL_CLASSES,
        MODEL_MAPPING,
        CLASSES_GROUPS,
        CVAT_LINK,
        # LOCAL_MODELS_DIR больше не нужен
    )
except ImportError:
    logger.critical(
        "Критическая ошибка: Не удалось импортировать конфигурацию из config.py."
    )
    exit()

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ---
IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}
SKIP_CLASSES_FOR_IOU_EVAL = {
    "safety",
    "no_safety",
    "chin_strap",
    "chin_strap_off",
    "glasses",
    "glasses_off",
}
INCLUSION_THRESHOLD_GT_COVERED = 0.8


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Без изменений) ---
def get_group_for_label(label: str) -> str:
    for group, labels in CLASSES_GROUPS.items():
        if label in labels:
            return group
    return label


def map_prediction_label_to_canonical(pred_label_from_model: str) -> str:
    for target, sources in MODEL_MAPPING.items():
        if pred_label_from_model in sources:
            return target
    return pred_label_from_model


def map_our_gt_label_to_model_label_set(our_gt_label: str) -> set:
    return OUR_TO_MODEL_CLASSES.get(our_gt_label, {our_gt_label})


def get_abs_bbox_from_normalized(norm_bbox, img_width, img_height):
    x, y, w, h_norm = norm_bbox
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h_norm) * img_height
    return [x1, y1, x2, y2]


def calculate_area(bbox_abs):
    if None in bbox_abs or bbox_abs[2] < bbox_abs[0] or bbox_abs[3] < bbox_abs[1]:
        return 0.0
    return (bbox_abs[2] - bbox_abs[0]) * (bbox_abs[3] - bbox_abs[1])


def calculate_intersection_area(bbox1_abs, bbox2_abs):
    x_left = max(bbox1_abs[0], bbox2_abs[0])
    y_top = max(bbox1_abs[1], bbox2_abs[1])
    x_right = min(bbox1_abs[2], bbox2_abs[2])
    y_bottom = min(bbox1_abs[3], bbox2_abs[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


# --- КАСТОМНАЯ ОЦЕНКА ПО ВХОЖДЕНИЮ (Без существенных изменений в логике, только логи) ---
def evaluate_by_inclusion(
    dataset, gt_field="ground_truth", pred_field="predictions", gt_covered_threshold=0.8
):
    logger.info(f"Оценка по вхождению для '{dataset.name}'...")
    view = dataset.view()
    for sample in view.iter_samples(autosave=True, progress=True):
        if sample.metadata is None or sample.metadata.width is None:
            try:
                sample.compute_metadata(overwrite=False)
            except Exception:
                logger.warning(
                    f"Пропуск сэмпла в evaluate_by_inclusion (нет метаданных): {sample.filepath}"
                )
                continue
            if sample.metadata is None or sample.metadata.width is None:
                logger.warning(
                    f"Пропуск сэмпла в evaluate_by_inclusion (нет метаданных): {sample.filepath}"
                )
                continue
        img_w, img_h = sample.metadata.width, sample.metadata.height
        gts = (
            sample[gt_field].detections
            if sample[gt_field] and sample[gt_field].detections
            else []
        )
        preds = (
            sample[pred_field].detections
            if sample[pred_field] and sample[pred_field].detections
            else []
        )
        abs_gts = [
            {"d": gt, "bb": get_abs_bbox_from_normalized(gt.bounding_box, img_w, img_h)}
            for gt in gts
        ]
        abs_preds = [
            {"d": p, "bb": get_abs_bbox_from_normalized(p.bounding_box, img_w, img_h)}
            for p in preds
        ]
        for gt_item in abs_gts:
            gt_d, gt_bb = gt_item["d"], gt_item["bb"]
            gt_area = calculate_area(gt_bb)
            gt_d["max_pred_inclusion_in_gt"] = 0.0
            gt_d["gt_covered_by_inclusion"] = False
            if gt_area > 0 and abs_preds:
                for pred_item in abs_preds:
                    incl = calculate_intersection_area(gt_bb, pred_item["bb"]) / gt_area
                    if incl > gt_d["max_pred_inclusion_in_gt"]:
                        gt_d["max_pred_inclusion_in_gt"] = incl
                if gt_d["max_pred_inclusion_in_gt"] >= gt_covered_threshold:
                    gt_d["gt_covered_by_inclusion"] = True
        for pred_item in abs_preds:
            pred_d, pred_bb = pred_item["d"], pred_item["bb"]
            pred_area = calculate_area(pred_bb)
            pred_d["max_gt_inclusion_in_pred"] = 0.0
            pred_d["max_gt_coverage_by_pred"] = 0.0
            if pred_area > 0 and abs_gts:
                for gt_item in abs_gts:
                    inter = calculate_intersection_area(gt_item["bb"], pred_bb)
                    if (
                        pred_area > 0
                        and inter / pred_area > pred_d["max_gt_inclusion_in_pred"]
                    ):
                        pred_d["max_gt_inclusion_in_pred"] = inter / pred_area
                    gt_ac = calculate_area(gt_item["bb"])
                    if gt_ac > 0 and inter / gt_ac > pred_d["max_gt_coverage_by_pred"]:
                        pred_d["max_gt_coverage_by_pred"] = inter / gt_ac
    logger.info(f"Оценка по вхождению для '{dataset.name}' завершена.")


# --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ПАТЧЕЙ (Используем автоматическую загрузку из Zoo) ---
def compute_and_save_patch_embeddings(
    dataset_or_view,
    zoo_model_name,  # Имя модели из FiftyOne Zoo (например, "dinov2-vitb14-torch")
    patches_field="ground_truth",
    embeddings_storage_field="embeddings",
):
    final_embeddings_field = f"{patches_field}_{embeddings_storage_field}"
    if not dataset_or_view.has_sample_field(patches_field):
        logger.warning(
            f"Поле '{patches_field}' не найдено в '{dataset_or_view.name}'. Пропуск эмбеддингов."
        )
        return

    logger.info(
        f"Вычисление эмбеддингов для '{patches_field}' в '{dataset_or_view.name}' используя модель Zoo: '{zoo_model_name}'."
    )
    logger.info(f"Результат будет сохранен в поле: '{final_embeddings_field}'.")

    try:
        detections_exist = any(
            s[patches_field] and s[patches_field].detections
            for s in dataset_or_view.select_fields(f"{patches_field}.detections")
        )
        if not detections_exist:
            logger.info(
                f"В поле '{patches_field}' датасета '{dataset_or_view.name}' нет объектов. Пропуск эмбеддингов."
            )
            return

        # Используем fo.compute_embeddings с именем модели из Zoo.
        # FiftyOne/PyTorch Hub должны сами справиться с загрузкой.
        dataset_or_view.compute_embeddings(
            zoo_model_name,  # Передаем имя модели из Zoo
            embeddings_field=final_embeddings_field,
            patches_field=patches_field,
            batch_size=16,
        )

        logger.info(f"Эмбеддинги успешно вычислены для '{dataset_or_view.name}'.")
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()
    except Exception as e:
        logger.exception(
            f"Критическая ошибка при вычислении эмбеддингов для '{dataset_or_view.name}' с моделью '{zoo_model_name}'."
        )
        logger.error(
            "Убедитесь, что имя модели ('{zoo_model_name}') корректно для вашего FiftyOne Zoo, "
            "есть интернет-соединение для загрузки, и все зависимости (torch, transformers, fiftyone-brain) установлены."
        )


# --- ЗАГРУЗКА ДАННЫХ И СОЗДАНИЕ ДАТАСЕТА ДЛЯ КЛАССА (Минимальные изменения в логике, только логи) ---
def load_class_dataset_from_csv(csv_file, all_predictions_dict, config_params):
    our_class_name_from_csv = os.path.splitext(os.path.basename(csv_file))[0]
    logger.info(
        f"Обработка: {our_class_name_from_csv} (из {os.path.basename(csv_file)})"
    )
    dataset_name = our_class_name_from_csv
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)  # Молча удаляем, если существует
    dataset = fo.Dataset(dataset_name, persistent=True)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Ошибка чтения {csv_file}: {e}")
        return None
    df = df.dropna(subset=["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"])
    req_cols = {
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "instance_label",
        "task_id",
        "job_id",
        "image_id",
    }
    if not req_cols.issubset(df.columns):
        logger.error(
            f"CSV {csv_file} не содержит колонки: {req_cols - set(df.columns)}. Пропуск."
        )
        return None

    target_model_labels = map_our_gt_label_to_model_label_set(our_class_name_from_csv)
    display_gt_label = (
        list(target_model_labels)[0] if target_model_labels else our_class_name_from_csv
    )

    img_data = {}
    for _, r in df.iterrows():
        img_p, img_n, gt_l = r["image_path"], r["image_name"], r["instance_label"]
        if gt_l != our_class_name_from_csv or not os.path.exists(img_p):
            continue
        try:
            w, h = int(r["image_width"]), int(r["image_height"])
            if w <= 0 or h <= 0:
                raise ValueError()
        except:
            logger.warning(
                f"Размеры для {img_n} в {csv_file} некорректны. Пропуск строки."
            )
            continue

        if img_p not in img_data:
            img_data[img_p] = {"gts": [], "preds": [], "s": (w, h), "n": img_n}
        x1, y1, x2, y2 = r["bbox_x_tl"], r["bbox_y_tl"], r["bbox_x_br"], r["bbox_y_br"]
        img_data[img_p]["gts"].append(
            {
                "label": display_gt_label,
                "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                "box_width_abs": float(x2 - x1),
                "box_height_abs": float(y2 - y1),
                "cvat_task": config_params["CVAT_LINK"]
                + f'/tasks/{r["task_id"]}/jobs/{int(r["job_id"])}?frame={r["image_id"]}',
            }
        )

    for img_p, data in img_data.items():
        w, h = data["s"]
        for pred_raw in all_predictions_dict.get(data["n"], []):
            if not all(k in pred_raw for k in ["label", "score", "bbox"]):
                continue
            pred_l_canon = map_prediction_label_to_canonical(pred_raw["label"])
            if pred_l_canon not in target_model_labels:
                continue
            grp = get_group_for_label(pred_l_canon)
            if grp in config_params["SIZE_REQUIREMENTS"]:
                mw, mh = config_params["SIZE_REQUIREMENTS"][grp]
                x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred_raw["bbox"]
                if (x2p_abs - x1p_abs) < mw or (y2p_abs - y1p_abs) < mh:
                    continue
            x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred_raw["bbox"]
            data["preds"].append(
                {
                    "label": pred_l_canon,
                    "bounding_box": [
                        x1p_abs / w,
                        y1p_abs / h,
                        (x2p_abs - x1p_abs) / w,
                        (y2p_abs - y1p_abs) / h,
                    ],
                    "confidence": pred_raw["score"],
                    "box_width_abs": float(x2p_abs - x1p_abs),
                    "box_height_abs": float(y2p_abs - y1p_abs),
                }
            )

    samples = [
        fo.Sample(
            filepath=fp,
            ground_truth=fo.Detections(
                detections=[fo.Detection(**gt_d) for gt_d in data["gts"]]
            ),
            predictions=fo.Detections(
                detections=[fo.Detection(**pred_d) for pred_d in data["preds"]]
            ),
        )
        for fp, data in img_data.items()
        if data["gts"]
    ]

    if not samples:
        logger.warning(f"Нет данных для '{dataset_name}'.")
        return dataset
    dataset.add_samples(samples)
    logger.info(f"Добавлено {len(samples)} сэмплов в '{dataset_name}'.")

    if our_class_name_from_csv in config_params["SKIP_CLASSES_FOR_IOU_EVAL"]:
        evaluate_by_inclusion(
            dataset,
            gt_covered_threshold=config_params["INCLUSION_THRESHOLD_GT_COVERED"],
        )
    else:
        if dataset.count("ground_truth.detections") > 0:
            for iou_thr, iou_tag in config_params["IOU_DICT"].items():
                try:
                    dataset.evaluate_detections(
                        "predictions",
                        gt_field="ground_truth",
                        eval_key=iou_tag,
                        method="coco",
                        iou=iou_thr,
                        compute_mAP=False,
                        classes=[display_gt_label],
                        progress=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка оценки IoU={iou_thr} для '{our_class_name_from_csv}': {e}"
                    )
        else:
            logger.warning(
                f"Пропуск оценки IoU для '{our_class_name_from_csv}': нет GT."
            )

    if config_params["COMPUTE_GT_EMBEDDINGS"]:
        logger.info(f"Вычисление эмбеддингов GT для '{dataset_name}'...")
        compute_and_save_patch_embeddings(
            dataset,
            zoo_model_name=config_params["ZOO_MODEL_NAME_FOR_EMBEDDINGS"],
            patches_field="ground_truth",
            embeddings_storage_field=config_params["EMBEDDINGS_FIELD_SUFFIX"],
        )

    logger.info(f"Обработка '{dataset_name}' завершена.")
    if config_params["LAUNCH_APP_FOR_EACH"] and samples:
        try:
            fo.launch_app(
                dataset,
                address="0.0.0.0",
                port=config_params["current_port"],
                auto=False,
            )
            logger.info(
                f"App для '{dataset_name}' доступен на порту {config_params['current_port']}"
            )
        except Exception as e:
            logger.error(f"Не удалось запустить App для '{dataset_name}': {e}")
    return dataset


# --- ОСНОВНАЯ ФУНКЦИЯ ---
def main():
    APP_CONFIG = {
        "LAUNCH_APP_FOR_EACH": False,
        "COMPUTE_GT_EMBEDDINGS": True,
        # Имя модели из FiftyOne Zoo для эмбеддингов.
        # Убедитесь, что оно есть в выводе `print(foz.list_downloadable_models(tags="embedding"))`
        "ZOO_MODEL_NAME_FOR_EMBEDDINGS": "dinov2-vitb14-torch",
        # "ZOO_MODEL_NAME_FOR_EMBEDDINGS": "clip-vit-base32-torch", # Альтернатива
        "EMBEDDINGS_FIELD_SUFFIX": "embeddings",  # Простой суффикс
        "START_PORT": 30082,
    }

    # PATH_TO_EMBEDDINGS_MODEL и EMBEDDINGS_MODEL_SUBDIR больше не нужны, т.к. модель грузится из Zoo

    LOADER_PARAMS = {
        "CVAT_LINK": CVAT_LINK,
        "SIZE_REQUIREMENTS": SIZE_REQUIREMENTS,
        "SKIP_CLASSES_FOR_IOU_EVAL": SKIP_CLASSES_FOR_IOU_EVAL,
        "INCLUSION_THRESHOLD_GT_COVERED": INCLUSION_THRESHOLD_GT_COVERED,
        "IOU_DICT": IOU_DICT,
        "COMPUTE_GT_EMBEDDINGS": APP_CONFIG["COMPUTE_GT_EMBEDDINGS"],
        "ZOO_MODEL_NAME_FOR_EMBEDDINGS": APP_CONFIG[
            "ZOO_MODEL_NAME_FOR_EMBEDDINGS"
        ],  # Передаем имя модели Zoo
        "EMBEDDINGS_FIELD_SUFFIX": APP_CONFIG["EMBEDDINGS_FIELD_SUFFIX"],
        "LAUNCH_APP_FOR_EACH": APP_CONFIG["LAUNCH_APP_FOR_EACH"],
    }

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            all_preds = json.load(f)
    except FileNotFoundError:
        logger.critical(f"Файл предсказаний не найден: {PATH_TO_PREDICTIONS}")
        return
    except json.JSONDecodeError:
        logger.critical(f"Ошибка декодирования JSON: {PATH_TO_PREDICTIONS}")
        return

    csv_files = sorted(glob.glob(os.path.join(PATH_TO_SPLIT, "*.csv")))
    if not csv_files:
        logger.warning(f"CSV файлы не найдены в {PATH_TO_SPLIT}")
        return

    loaded_datasets_summary = []
    for i, csv_f in enumerate(csv_files):
        LOADER_PARAMS["current_port"] = APP_CONFIG["START_PORT"] + i
        ds = load_class_dataset_from_csv(csv_f, all_preds, LOADER_PARAMS)
        if ds:
            info = {"name": ds.name, "emb": False}
            if APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] and ds.has_field(
                f"ground_truth_{APP_CONFIG['EMBEDDINGS_FIELD_SUFFIX']}"
            ):
                info["emb"] = True
            if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and ds.count() > 0:
                info["app_port"] = LOADER_PARAMS["current_port"]
            loaded_datasets_summary.append(info)
            # if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and ds and ds.count() > 0: # Убрал ожидание input

    logger.info("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_summary:
        logger.info("Созданы/обновлены датасеты:")
        for info in loaded_datasets_summary:
            msg = (
                f"- {info['name']}"
                + (" (с эмб. GT)" if info["emb"] else "")
                + (f" (App на порту {info['app_port']})" if "app_port" in info else "")
            )
            logger.info(msg)
        if loaded_datasets_summary:
            logger.info(f"\nДля просмотра (если App не был запущен):")
            logger.info(f"import fiftyone as fo")
            logger.info(
                f"dataset = fo.load_dataset('{loaded_datasets_summary[0]['name']}')"
            )
            logger.info(
                f"session = fo.launch_app(dataset, port={APP_CONFIG['START_PORT']})"
            )
    else:
        logger.warning("Не было создано ни одного датасета.")


if __name__ == "__main__":
    # fo.config.show_progress_bars = False # Можно раскомментировать, чтобы убрать прогресс-бары FO
    main()
