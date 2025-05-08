import os
import glob
import json
import pandas as pd
import numpy as np  # Используется неявно fiftyone или sklearn
import fiftyone as fo
import fiftyone.zoo as foz  # Используется неявно для compute_embeddings
from loguru import logger

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
# logger.add("processing_{time}.log", level="DEBUG", rotation="10 MB") # Опционально: запись в файл с ротацией

# --- ИМПОРТ КОНФИГУРАЦИИ ---
# Предполагается, что эти переменные определены в config.py
# Пример содержимого config.py смотрите в README
try:
    from config import (
        PATH_TO_SPLIT,
        PATH_TO_PREDICTIONS,
        SIZE_REQUIREMENTS,
        OUR_TO_MODEL_CLASSES,
        MODEL_MAPPING,
        CLASSES_GROUPS,
        CVAT_LINK,
        LOCAL_MODELS_DIR,  # Путь к директории, где хранятся скачанные модели
    )
except ImportError:
    logger.critical(
        "Не удалось импортировать конфигурацию из config.py. Убедитесь, что файл существует и содержит все необходимые переменные."
    )
    exit()

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ИЗ СКРИПТА (можно вынести в config.py) ---
IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}
# Классы, для которых используется кастомная оценка по вхождению вместо IoU
SKIP_CLASSES_FOR_IOU_EVAL = {
    "safety",
    "no_safety",
    "chin_strap",
    "chin_strap_off",
    "glasses",
    "glasses_off",
}
INCLUSION_THRESHOLD_GT_COVERED = 0.8  # Порог для кастомной метрики вхождения


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_group_for_label(label: str) -> str:
    for group, labels in CLASSES_GROUPS.items():
        if label in labels:
            return group
    return label


def map_prediction_label_to_canonical(pred_label_from_model: str) -> str:
    """Маппинг метки предсказания модели к канонической метке для сравнения."""
    for target_label, source_model_labels in MODEL_MAPPING.items():
        if pred_label_from_model in source_model_labels:
            return target_label
    return pred_label_from_model


def map_our_gt_label_to_model_label_set(our_gt_label: str) -> set:
    """
    Преобразование нашей GT метки (из CSV) в НАБОР соответствующих меток модели.
    Это используется для фильтрации GT и предсказаний для конкретного датасета класса.
    """
    return OUR_TO_MODEL_CLASSES.get(our_gt_label, {our_gt_label})


# --- ФУНКЦИИ ДЛЯ РАСЧЕТА ГЕОМЕТРИИ ---
def get_abs_bbox_from_normalized(norm_bbox, img_width, img_height):
    x, y, w, h = norm_bbox
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h) * img_height
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


# --- КАСТОМНАЯ ОЦЕНКА ПО ВХОЖДЕНИЮ ---
def evaluate_by_inclusion(
    dataset, gt_field="ground_truth", pred_field="predictions", gt_covered_threshold=0.8
):
    logger.info(f"Запуск кастомной оценки по вхождению для датасета: {dataset.name}")
    view = dataset.view()

    for sample in view.iter_samples(autosave=True, progress=True):
        if (
            sample.metadata is None
            or sample.metadata.width is None
            or sample.metadata.height is None
        ):
            try:
                sample.compute_metadata(overwrite=False)
                if sample.metadata is None or sample.metadata.width is None:
                    logger.error(
                        f"Не удалось вычислить метаданные для {sample.filepath}. Пропуск оценки вхождения для этого сэмпла."
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Ошибка вычисления метаданных для {sample.filepath}: {e}. Пропуск оценки вхождения."
                )
                continue

        img_width, img_height = sample.metadata.width, sample.metadata.height
        gt_detections = (
            sample[gt_field].detections
            if sample[gt_field] and sample[gt_field].detections
            else []
        )
        pred_detections = (
            sample[pred_field].detections
            if sample[pred_field] and sample[pred_field].detections
            else []
        )

        abs_gts = [
            {
                "det": gt,
                "abs_bbox": get_abs_bbox_from_normalized(
                    gt.bounding_box, img_width, img_height
                ),
            }
            for gt in gt_detections
        ]
        abs_preds = [
            {
                "det": p,
                "abs_bbox": get_abs_bbox_from_normalized(
                    p.bounding_box, img_width, img_height
                ),
            }
            for p in pred_detections
        ]

        for gt_item in abs_gts:
            gt_det, gt_abs_bbox = gt_item["det"], gt_item["abs_bbox"]
            gt_area = calculate_area(gt_abs_bbox)
            gt_det["max_pred_inclusion_in_gt"] = 0.0
            gt_det["gt_covered_by_inclusion"] = False
            if gt_area > 0 and abs_preds:
                for pred_item in abs_preds:
                    intersection = calculate_intersection_area(
                        gt_abs_bbox, pred_item["abs_bbox"]
                    )
                    inclusion = intersection / gt_area
                    if inclusion > gt_det["max_pred_inclusion_in_gt"]:
                        gt_det["max_pred_inclusion_in_gt"] = inclusion
                if gt_det["max_pred_inclusion_in_gt"] >= gt_covered_threshold:
                    gt_det["gt_covered_by_inclusion"] = True

        for pred_item in abs_preds:
            pred_det, pred_abs_bbox = pred_item["det"], pred_item["abs_bbox"]
            pred_area = calculate_area(pred_abs_bbox)
            pred_det["max_gt_inclusion_in_pred"] = 0.0  # Area(I(GT,P)) / Area(P)
            pred_det["max_gt_coverage_by_pred"] = (
                0.0  # Area(I(GT,P)) / Area(GT) - насколько этот пред покрывает какой-либо GT
            )
            if pred_area > 0 and abs_gts:
                for gt_item in abs_gts:
                    intersection = calculate_intersection_area(
                        gt_item["abs_bbox"], pred_abs_bbox
                    )
                    inclusion_in_p = intersection / pred_area
                    if inclusion_in_p > pred_det["max_gt_inclusion_in_pred"]:
                        pred_det["max_gt_inclusion_in_pred"] = inclusion_in_p

                    gt_area_for_coverage = calculate_area(gt_item["abs_bbox"])
                    if gt_area_for_coverage > 0:
                        coverage_of_gt = intersection / gt_area_for_coverage
                        if coverage_of_gt > pred_det["max_gt_coverage_by_pred"]:
                            pred_det["max_gt_coverage_by_pred"] = coverage_of_gt
    logger.info(f"Кастомная оценка по вхождению для {dataset.name} завершена.")


# --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ПАТЧЕЙ ---
def compute_and_save_patch_embeddings(
    dataset_or_view,
    model_path,
    patches_field="ground_truth",
    embeddings_storage_field="clip_embeddings",
):
    final_embeddings_field = f"{patches_field}_{embeddings_storage_field}"
    if not dataset_or_view.has_sample_field(patches_field):
        logger.warning(
            f"Поле '{patches_field}' не найдено в {dataset_or_view.name}. Пропуск вычисления эмбеддингов."
        )
        return

    logger.info(
        f"Вычисление эмбеддингов для объектов из '{patches_field}' датасета '{dataset_or_view.name}'."
    )
    logger.info(
        f"Путь к модели: {model_path}. Результат в поле: '{final_embeddings_field}'."
    )

    try:
        # Проверка наличия объектов перед вызовом compute_embeddings
        detections_exist = any(
            sample[patches_field] and sample[patches_field].detections
            for sample in dataset_or_view.select_fields(f"{patches_field}.detections")
        )
        if not detections_exist:
            logger.info(
                f"В поле '{patches_field}' датасета '{dataset_or_view.name}' нет объектов. Пропуск."
            )
            return

        fo.compute_embeddings(
            dataset_or_view,
            model_path,
            embeddings_field=final_embeddings_field,
            patches_field=patches_field,
        )
        logger.info(f"Эмбеддинги успешно вычислены для {dataset_or_view.name}.")
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()
    except Exception as e:
        logger.error(
            f"Ошибка при вычислении эмбеддингов для {dataset_or_view.name} с моделью {model_path}: {e}"
        )
        logger.error(
            "Убедитесь, что путь к модели корректен и содержит все необходимые файлы, а также установлены зависимости (torch, transformers)."
        )


# --- ЗАГРУЗКА ДАННЫХ И СОЗДАНИЕ ДАТАСЕТА ДЛЯ КЛАССА ---
def load_class_dataset_from_csv(csv_file, all_predictions_dict, config_params):
    our_class_name_from_csv = os.path.splitext(os.path.basename(csv_file))[0]
    logger.info(
        f"=== Обработка CSV: {csv_file} (Наш класс: {our_class_name_from_csv}) ==="
    )

    dataset_name = our_class_name_from_csv
    if dataset_name in fo.list_datasets():
        logger.info(
            f"Датасет {dataset_name} уже существует. Удаление и создание заново."
        )
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name, persistent=True)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Не удалось прочитать CSV файл {csv_file}: {e}")
        return None
    df = df.dropna(subset=["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"])

    required_cols = {
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "instance_label",
    }
    if not required_cols.issubset(df.columns):
        logger.error(
            f"CSV {csv_file} должен содержать столбцы: {required_cols}. Пропуск файла."
        )
        return None

    # Метки модели, соответствующие текущему нашему классу из CSV
    target_model_labels_for_csv = map_our_gt_label_to_model_label_set(
        our_class_name_from_csv
    )
    if not target_model_labels_for_csv:
        logger.warning(
            f"Для нашего класса '{our_class_name_from_csv}' нет соответствий в классах модели (OUR_TO_MODEL_CLASSES)."
        )
    # Метка для отображения GT в датасете (обычно первая из target_model_labels_for_csv)
    display_gt_label_for_dataset = (
        list(target_model_labels_for_csv)[0]
        if target_model_labels_for_csv
        else our_class_name_from_csv
    )

    processed_image_data = (
        {}
    )  # {image_path: {"gt": [], "pred": [], "size": (), "name": ""}}

    for _, row in df.iterrows():
        image_path, image_name = row["image_path"], row["image_name"]
        gt_label_our = row["instance_label"]

        if gt_label_our != our_class_name_from_csv:  # Только GT текущего класса из CSV
            continue
        if not os.path.exists(image_path):
            logger.warning(
                f"Файл изображения {image_path} из {csv_file} не найден. Пропуск строки."
            )
            continue
        try:
            w, h = int(row["image_width"]), int(row["image_height"])
            if w <= 0 or h <= 0:
                raise ValueError("Incorrect image dimensions")
        except ValueError:
            logger.warning(
                f"Некорректные размеры изображения для {image_name} в {csv_file}. Пропуск строки."
            )
            continue

        if image_path not in processed_image_data:
            processed_image_data[image_path] = {
                "gt_detections": [],
                "pred_detections": [],
                "size": (w, h),
                "image_name": image_name,
            }

        x1, y1, x2, y2 = (
            row["bbox_x_tl"],
            row["bbox_y_tl"],
            row["bbox_x_br"],
            row["bbox_y_br"],
        )
        gt_data = {
            "label": display_gt_label_for_dataset,
            "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            "box_width_abs": float(x2 - x1),
            "box_height_abs": float(y2 - y1),
            "cvat_task": config_params["CVAT_LINK"]
            + f'/tasks/{row["task_id"]}/jobs/{int(row["job_id"])}?frame={row["image_id"]}',
        }
        processed_image_data[image_path]["gt_detections"].append(gt_data)

    for image_path, data_for_image in processed_image_data.items():
        w, h = data_for_image["size"]
        preds_on_image = all_predictions_dict.get(data_for_image["image_name"], [])

        for pred_raw in preds_on_image:
            if not all(k in pred_raw for k in ["label", "score", "bbox"]):
                continue

            pred_label_canonical = map_prediction_label_to_canonical(pred_raw["label"])
            if (
                pred_label_canonical not in target_model_labels_for_csv
            ):  # Только предсказания релевантных классов
                continue

            group = get_group_for_label(pred_label_canonical)
            if group in config_params["SIZE_REQUIREMENTS"]:
                min_w, min_h = config_params["SIZE_REQUIREMENTS"][group]
                x1p, y1p, x2p, y2p = pred_raw[
                    "bbox"
                ]  # Предполагаем абсолютные координаты
                if (x2p - x1p) < min_w or (y2p - y1p) < min_h:
                    continue

            x1p, y1p, x2p, y2p = pred_raw["bbox"]
            pred_data = {
                "label": pred_label_canonical,
                "bounding_box": [x1p / w, y1p / h, (x2p - x1p) / w, (y2p - y1p) / h],
                "confidence": pred_raw["score"],
                "box_width_abs": float(x2p - x1p),
                "box_height_abs": float(y2p - y1p),
            }
            data_for_image["pred_detections"].append(pred_data)

    samples_to_add = [
        fo.Sample(
            filepath=fp,
            ground_truth=fo.Detections(
                detections=[fo.Detection(**d) for d in data["gt_detections"]]
            ),
            predictions=fo.Detections(
                detections=[fo.Detection(**d) for d in data["pred_detections"]]
            ),
        )
        for fp, data in processed_image_data.items()
        if data["gt_detections"]  # Только если есть GT для этого класса
    ]

    if samples_to_add:
        dataset.add_samples(samples_to_add)
        logger.info(
            f"Добавлено {len(samples_to_add)} сэмплов в датасет {dataset_name}."
        )
    else:
        logger.warning(
            f"Нет валидных данных (с GT для класса {our_class_name_from_csv}) для датасета {dataset_name}."
        )
        return dataset  # Возвращаем, возможно, пустой датасет

    # --- ОЦЕНКА ---
    if our_class_name_from_csv in config_params["SKIP_CLASSES_FOR_IOU_EVAL"]:
        evaluate_by_inclusion(
            dataset,
            gt_covered_threshold=config_params["INCLUSION_THRESHOLD_GT_COVERED"],
        )
    else:
        logger.info(
            f"Класс {our_class_name_from_csv} НЕ в SKIP. Запуск стандартной оценки IoU."
        )
        if dataset.count(f"ground_truth.detections") > 0:
            for iou_thr, iou_tag in config_params["IOU_DICT"].items():
                logger.info(
                    f"Оценка для IoU={iou_thr} (ключ:{iou_tag}), класс для оценки: {display_gt_label_for_dataset}"
                )
                try:
                    dataset.evaluate_detections(
                        "predictions",
                        gt_field="ground_truth",
                        eval_key=iou_tag,
                        method="coco",
                        iou=iou_thr,
                        compute_mAP=False,
                        classes=[display_gt_label_for_dataset],
                        progress=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка оценки IoU={iou_thr} для {our_class_name_from_csv} (метка {display_gt_label_for_dataset}): {e}"
                    )
        else:
            logger.warning(
                f"Пропуск оценки IoU для {our_class_name_from_csv}: нет GT детекций."
            )

    # --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ---
    if config_params["COMPUTE_GT_EMBEDDINGS"]:
        if config_params["PATH_TO_EMBEDDINGS_MODEL"] and os.path.isdir(
            config_params["PATH_TO_EMBEDDINGS_MODEL"]
        ):
            logger.info(
                f"Вычисление эмбеддингов для GT объектов датасета {dataset_name}..."
            )
            compute_and_save_patch_embeddings(
                dataset,
                model_path=config_params["PATH_TO_EMBEDDINGS_MODEL"],
                patches_field="ground_truth",
                embeddings_storage_field=config_params["EMBEDDINGS_FIELD_SUFFIX"],
            )
        else:
            logger.warning(
                f"Путь к модели эмбеддингов не указан/некорректен: {config_params['PATH_TO_EMBEDDINGS_MODEL']}. Пропуск."
            )

    logger.info(f"Датасет {dataset_name} обработан и сохранен.")
    if config_params["LAUNCH_APP_FOR_EACH"] and samples_to_add:
        logger.info(
            f"Запуск FiftyOne App для {dataset_name} на порту {config_params['current_port']}..."
        )
        try:
            fo.launch_app(
                dataset,
                address="0.0.0.0",
                port=config_params["current_port"],
                auto=False,
            )
            logger.info(
                f"FiftyOne App доступен: http://<ваш_ip>:{config_params['current_port']}"
            )
        except Exception as e:
            logger.error(f"Не удалось запустить FiftyOne App для {dataset_name}: {e}")
    return dataset


# --- ОСНОВНАЯ ФУНКЦИЯ ---
def main():
    # --- Настройки выполнения ---
    # (Эти параметры можно также вынести в config.py или передавать через CLI)
    APP_CONFIG = {
        "LAUNCH_APP_FOR_EACH": False,  # Открывать ли браузер для каждого датасета
        "COMPUTE_GT_EMBEDDINGS": True,  # Вычислять ли эмбеддинги для GT патчей
        "EMBEDDINGS_MODEL_SUBDIR": "clip-vit-base-patch32",  # Поддиректория модели в LOCAL_MODELS_DIR
        "EMBEDDINGS_FIELD_SUFFIX": "clip_embeddings",  # Суффикс для поля с эмбеддингами
        "START_PORT": 30082,  # Начальный порт для FiftyOne App
    }
    APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"] = os.path.join(
        LOCAL_MODELS_DIR, APP_CONFIG["EMBEDDINGS_MODEL_SUBDIR"]
    )

    if APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] and not os.path.isdir(
        APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"]
    ):
        logger.warning(
            f"Директория с локальной моделью для эмбеддингов не найдена: {APP_CONFIG['PATH_TO_EMBEDDINGS_MODEL']}"
        )
        logger.warning("Вычисление эмбеддингов будет пропущено.")
        APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] = False

    # Передача глобальных констант и настроек в функцию обработки
    # чтобы избежать использования global или слишком много параметров
    SCRIPT_PARAMS_FOR_LOADER = {
        "CVAT_LINK": CVAT_LINK,
        "SIZE_REQUIREMENTS": SIZE_REQUIREMENTS,
        "SKIP_CLASSES_FOR_IOU_EVAL": SKIP_CLASSES_FOR_IOU_EVAL,
        "INCLUSION_THRESHOLD_GT_COVERED": INCLUSION_THRESHOLD_GT_COVERED,
        "IOU_DICT": IOU_DICT,
        "COMPUTE_GT_EMBEDDINGS": APP_CONFIG["COMPUTE_GT_EMBEDDINGS"],
        "PATH_TO_EMBEDDINGS_MODEL": APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"],
        "EMBEDDINGS_FIELD_SUFFIX": APP_CONFIG["EMBEDDINGS_FIELD_SUFFIX"],
        "LAUNCH_APP_FOR_EACH": APP_CONFIG["LAUNCH_APP_FOR_EACH"],
        # current_port будет добавляться в цикле
    }

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            all_predictions_dict = json.load(f)
    except FileNotFoundError:
        logger.critical(f"Файл с предсказаниями не найден: {PATH_TO_PREDICTIONS}")
        return
    except json.JSONDecodeError:
        logger.critical(f"Не удалось декодировать JSON из файла: {PATH_TO_PREDICTIONS}")
        return

    csv_files = sorted(glob.glob(os.path.join(PATH_TO_SPLIT, "*.csv")))
    if not csv_files:
        logger.warning(f"CSV файлы не найдены в {PATH_TO_SPLIT}")
        return

    loaded_datasets_summary = []
    for i, csv_file in enumerate(csv_files):
        SCRIPT_PARAMS_FOR_LOADER["current_port"] = APP_CONFIG["START_PORT"] + i

        dataset = load_class_dataset_from_csv(
            csv_file, all_predictions_dict, SCRIPT_PARAMS_FOR_LOADER
        )
        if dataset:
            summary = {"name": dataset.name, "has_embeddings": False}
            if APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] and dataset.has_field(
                f"ground_truth_{APP_CONFIG['EMBEDDINGS_FIELD_SUFFIX']}"
            ):
                summary["has_embeddings"] = True
            if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and dataset.count() > 0:
                summary["app_launched_port"] = SCRIPT_PARAMS_FOR_LOADER["current_port"]
            loaded_datasets_summary.append(summary)
            if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and dataset and dataset.count() > 0:
                logger.info(
                    f"Обработка {dataset.name} завершена. Для продолжения закройте вкладку/консоль FiftyOne или нажмите Ctrl+C, если скрипт ожидает."
                )
                # input("Нажмите Enter для следующего файла...") # Для пошагового режима

    logger.info("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_summary:
        logger.info("Созданы/обновлены датасеты:")
        for info in loaded_datasets_summary:
            msg = f"- {info['name']}"
            if info["has_embeddings"]:
                msg += " (с эмбеддингами GT)"
            if "app_launched_port" in info:
                msg += f" (App запущен на порту {info['app_launched_port']})"
            logger.info(msg)

        logger.info("\nДля просмотра датасета вручную (если App не был запущен):")
        logger.info("import fiftyone as fo")
        logger.info(
            f"dataset = fo.load_dataset('{loaded_datasets_summary[0]['name']}') # Замените на имя нужного датасета"
        )
        logger.info(
            f"session = fo.launch_app(dataset, port={APP_CONFIG['START_PORT']})"
        )
    else:
        logger.warning("Не было создано ни одного датасета.")


if __name__ == "__main__":
    # fo.config.show_progress_bars = False # Отключить прогресс-бары FiftyOne (если мешают логам)
    main()
