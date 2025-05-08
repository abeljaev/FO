import os
import glob
import json
import pandas as pd
import numpy as np
import fiftyone as fo

# import fiftyone.zoo as foz # foz может больше не понадобиться, если не используем list_zoo_models
from loguru import logger
from PIL import Image
import torch

# from torchvision import transforms as T # T может не понадобиться, если AutoImageProcessor справляется
from transformers import AutoImageProcessor, AutoModel

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
# logger.add("processing_{time}.log", level="DEBUG", rotation="10 MB")

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
        LOCAL_MODELS_DIR,
    )
except ImportError:
    logger.critical(
        "Не удалось импортировать конфигурацию из config.py. Убедитесь, что файл существует и содержит все необходимые переменные, включая LOCAL_MODELS_DIR."
    )
    exit()

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ ИЗ СКРИПТА ---
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


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
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


# --- ФУНКЦИИ ДЛЯ РАСЧЕТА ГЕОМЕТРИИ ---
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
                        f"Не удалось вычислить метаданные для {sample.filepath}. Пропуск."
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Ошибка вычисления метаданных для {sample.filepath}: {e}. Пропуск."
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
                    gt_area_curr = calculate_area(gt_item["bb"])
                    if (
                        gt_area_curr > 0
                        and inter / gt_area_curr > pred_d["max_gt_coverage_by_pred"]
                    ):
                        pred_d["max_gt_coverage_by_pred"] = inter / gt_area_curr
    logger.info(f"Кастомная оценка по вхождению для {dataset.name} завершена.")


# --- КЛАСС-ОБЕРТКА ДЛЯ ЛОКАЛЬНОЙ МОДЕЛИ HUGGING FACE ---
class HuggingFaceEmbedder(fo.core.models.Model):
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Загрузка модели Hugging Face из: {self.model_path}")
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        logger.info(f"Модель загружена на устройство: {self.device}")
        self._embedding_dim = getattr(self.model.config, "hidden_size", 768)

    @property
    def has_detector(self):
        return False

    @property
    def has_embedder(self):
        return True

    @property
    def media_type(self):
        return "image"

    def get_embeddings(self, frames_or_images):
        embeddings = []
        with torch.no_grad():
            for img_or_path in frames_or_images:
                try:
                    if isinstance(img_or_path, str):
                        img = Image.open(img_or_path).convert("RGB")
                    elif isinstance(img_or_path, np.ndarray):
                        img = Image.fromarray(img_or_path).convert("RGB")
                    else:
                        raise TypeError(
                            f"Неподдерживаемый тип входных данных: {type(img_or_path)}"
                        )

                    inputs = self.processor(images=img, return_tensors="pt").to(
                        self.device
                    )
                    outputs = self.model(**inputs)

                    if hasattr(outputs, "last_hidden_state"):
                        emb = (
                            outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                        )  # CLS token
                    elif hasattr(outputs, "pooler_output"):
                        emb = outputs.pooler_output.cpu().numpy().squeeze()
                    else:
                        logger.warning(
                            f"Не удалось стандартно извлечь эмбеддинг для {self.model_path}. Проверьте структуру выхода модели."
                        )
                        # Попытка взять первый элемент, если это тензор или список/кортеж тензоров
                        output_data = (
                            outputs[0]
                            if isinstance(outputs, (list, tuple))
                            else outputs
                        )
                        if (
                            isinstance(output_data, torch.Tensor)
                            and output_data.ndim >= 2
                        ):
                            # Предполагаем [batch, seq_len, hidden_dim] и берем CLS или [batch, hidden_dim]
                            emb = (
                                output_data[:, 0, :].cpu().numpy().squeeze()
                                if output_data.ndim == 3
                                else output_data.cpu().numpy().squeeze()
                            )
                        else:
                            emb = np.zeros(self._embedding_dim, dtype=np.float32)

                except Exception as e:
                    logger.error(
                        f"Ошибка обработки изображения/получения эмбеддинга для '{str(img_or_path)[:50]}...': {e}"
                    )
                    emb = np.zeros(
                        self._embedding_dim, dtype=np.float32
                    )  # Заглушка в случае ошибки

                if emb.ndim == 0 or emb.shape[0] != self._embedding_dim:
                    logger.warning(
                        f"Получен эмбеддинг неожиданной формы {emb.shape if hasattr(emb, 'shape') else type(emb)} для '{str(img_or_path)[:50]}...'. Используется нулевой вектор."
                    )
                    emb = np.zeros(self._embedding_dim, dtype=np.float32)

                embeddings.append(emb)
        return np.array(embeddings)

    def embed_all(self, frames_or_images):  # FiftyOne <=0.21.x
        return self.get_embeddings(frames_or_images)

    def embed(self, frame_or_image):  # FiftyOne >=0.22.x
        return self.get_embeddings([frame_or_image])[0]


# --- ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ ПАТЧЕЙ ---
def compute_and_save_patch_embeddings(
    dataset_or_view,
    model_path,
    patches_field="ground_truth",
    embeddings_storage_field="embeddings",  # Будет передан из APP_CONFIG
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
    logger.info(f"Используется кастомная обертка HuggingFaceEmbedder.")

    try:
        detections_exist = any(
            sample[patches_field] and sample[patches_field].detections
            for sample in dataset_or_view.select_fields(f"{patches_field}.detections")
        )
        if not detections_exist:
            logger.info(
                f"В поле '{patches_field}' датасета '{dataset_or_view.name}' нет объектов. Пропуск."
            )
            return

        model_instance = HuggingFaceEmbedder(model_path=model_path)

        dataset_or_view.compute_embeddings(
            model_instance,
            embeddings_field=final_embeddings_field,
            patches_field=patches_field,
            batch_size=16,  # Можно настроить
        )

        logger.info(f"Эмбеддинги успешно вычислены для {dataset_or_view.name}.")
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()
    except Exception as e:
        logger.exception(
            f"Ошибка при вычислении эмбеддингов для {dataset_or_view.name} с моделью {model_path}"
        )
        logger.error(
            "Убедитесь, что путь к модели ('{model_path}') корректен и содержит все необходимые файлы. "
            "Проверьте зависимости (torch, transformers, PIL)."
        )


# --- ЗАГРУЗКА ДАННЫХ И СОЗДАНИЕ ДАТАСЕТА ДЛЯ КЛАССА ---
def load_class_dataset_from_csv(csv_file, all_predictions_dict, config_params):
    our_class_name_from_csv = os.path.splitext(os.path.basename(csv_file))[0]
    logger.info(
        f"=== Обработка CSV: {csv_file} (Наш класс: {our_class_name_from_csv}) ==="
    )
    dataset_name = our_class_name_from_csv
    if dataset_name in fo.list_datasets():
        logger.info(f"Датасет {dataset_name} уже существует. Удаление.")
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name, persistent=True)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Не удалось прочитать CSV {csv_file}: {e}")
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
        missing_cols = req_cols - set(df.columns)
        logger.error(
            f"CSV {csv_file} должен содержать {req_cols}. Отсутствуют: {missing_cols}. Пропуск."
        )
        return None

    target_model_labels = map_our_gt_label_to_model_label_set(our_class_name_from_csv)
    display_gt_label = (
        list(target_model_labels)[0] if target_model_labels else our_class_name_from_csv
    )

    img_data = {}
    for _, r in df.iterrows():
        img_p, img_n, gt_l = r["image_path"], r["image_name"], r["instance_label"]
        if gt_l != our_class_name_from_csv:
            continue
        if not os.path.exists(img_p):
            logger.warning(f"Файл {img_p} из {csv_file} не найден. Пропуск строки.")
            continue
        try:
            w, h = int(r["image_width"]), int(r["image_height"])
            if w <= 0 or h <= 0:
                raise ValueError("Incorrect image dimensions")
        except (ValueError, TypeError):
            logger.warning(
                f"Размеры для {img_n} в {csv_file} некорректны: w='{r['image_width']}', h='{r['image_height']}'. Пропуск."
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

    samples = []
    for fp, data in img_data.items():
        if not data["gts"]:
            continue
        gt_dets = [fo.Detection(**gt_data) for gt_data in data["gts"]]
        pred_dets = [fo.Detection(**pred_data) for pred_data in data["preds"]]
        samples.append(
            fo.Sample(
                filepath=fp,
                ground_truth=fo.Detections(detections=gt_dets),
                predictions=fo.Detections(detections=pred_dets),
            )
        )

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Добавлено {len(samples)} сэмплов в датасет {dataset_name}.")
    else:
        logger.warning(f"Нет валидных данных для датасета {dataset_name}.")
        return dataset

    if our_class_name_from_csv in config_params["SKIP_CLASSES_FOR_IOU_EVAL"]:
        evaluate_by_inclusion(
            dataset,
            gt_covered_threshold=config_params["INCLUSION_THRESHOLD_GT_COVERED"],
        )
    else:
        logger.info(
            f"Класс {our_class_name_from_csv} НЕ в SKIP. Стандартная оценка IoU."
        )
        if dataset.count("ground_truth.detections") > 0:
            for iou_thr, iou_tag in config_params["IOU_DICT"].items():
                logger.info(
                    f"Оценка IoU={iou_thr} (ключ:{iou_tag}), класс для оценки: {display_gt_label}"
                )
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
                        f"Ошибка оценки IoU={iou_thr} для {our_class_name_from_csv} ({display_gt_label}): {e}"
                    )
        else:
            logger.warning(
                f"Пропуск оценки IoU для {our_class_name_from_csv}: нет GT детекций."
            )

    if config_params["COMPUTE_GT_EMBEDDINGS"]:
        path_emb_model = config_params["PATH_TO_EMBEDDINGS_MODEL"]
        if path_emb_model and os.path.isdir(path_emb_model):
            logger.info(f"Вычисление эмбеддингов GT для {dataset_name}...")
            compute_and_save_patch_embeddings(
                dataset,
                model_path=path_emb_model,
                patches_field="ground_truth",
                embeddings_storage_field=config_params["EMBEDDINGS_FIELD_SUFFIX"],
                # zoo_model_name_for_type не передается, т.к. HuggingFaceEmbedder его не использует
            )
        else:
            logger.warning(
                f"Путь к модели эмбеддингов не указан/некорректен: {path_emb_model}. Пропуск."
            )

    logger.info(f"Датасет {dataset_name} обработан.")
    if config_params["LAUNCH_APP_FOR_EACH"] and samples:
        logger.info(
            f"Запуск App для {dataset_name} на порту {config_params['current_port']}..."
        )
        try:
            fo.launch_app(
                dataset,
                address="0.0.0.0",
                port=config_params["current_port"],
                auto=False,
            )
            logger.info(
                f"App доступен: http://<ваш_ip>:{config_params['current_port']}"
            )
        except Exception as e:
            logger.error(f"Не удалось запустить App для {dataset_name}: {e}")
    return dataset


# --- ОСНОВНАЯ ФУНКЦИЯ ---
def main():
    APP_CONFIG = {
        "LAUNCH_APP_FOR_EACH": False,
        "COMPUTE_GT_EMBEDDINGS": True,
        "EMBEDDINGS_MODEL_SUBDIR": "dinov2-vitb14",
        "EMBEDDINGS_FIELD_SUFFIX": "dinov2b14_embeddings",
        # "EMBEDDINGS_MODEL_SUBDIR": "clip-vit-base32",
        # "EMBEDDINGS_FIELD_SUFFIX": "clip_vitb32_embeddings",
        "START_PORT": 30082,
    }
    APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"] = os.path.join(
        LOCAL_MODELS_DIR, APP_CONFIG["EMBEDDINGS_MODEL_SUBDIR"]
    )

    if APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] and not os.path.isdir(
        APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"]
    ):
        logger.warning(
            f"Директория модели эмбеддингов не найдена: {APP_CONFIG['PATH_TO_EMBEDDINGS_MODEL']}. Эмбеддинги не будут вычислены."
        )
        APP_CONFIG["COMPUTE_GT_EMBEDDINGS"] = False

    LOADER_PARAMS = {
        "CVAT_LINK": CVAT_LINK,
        "SIZE_REQUIREMENTS": SIZE_REQUIREMENTS,
        "SKIP_CLASSES_FOR_IOU_EVAL": SKIP_CLASSES_FOR_IOU_EVAL,
        "INCLUSION_THRESHOLD_GT_COVERED": INCLUSION_THRESHOLD_GT_COVERED,
        "IOU_DICT": IOU_DICT,
        "COMPUTE_GT_EMBEDDINGS": APP_CONFIG["COMPUTE_GT_EMBEDDINGS"],
        "PATH_TO_EMBEDDINGS_MODEL": APP_CONFIG["PATH_TO_EMBEDDINGS_MODEL"],
        "EMBEDDINGS_FIELD_SUFFIX": APP_CONFIG["EMBEDDINGS_FIELD_SUFFIX"],
        "LAUNCH_APP_FOR_EACH": APP_CONFIG["LAUNCH_APP_FOR_EACH"],
    }

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            all_preds = json.load(f)
    except FileNotFoundError:
        logger.critical(f"Предсказания не найдены: {PATH_TO_PREDICTIONS}")
        return
    except json.JSONDecodeError:
        logger.critical(f"Ошибка декодирования JSON: {PATH_TO_PREDICTIONS}")
        return

    csv_files = sorted(glob.glob(os.path.join(PATH_TO_SPLIT, "*.csv")))
    if not csv_files:
        logger.warning(f"CSV не найдены в {PATH_TO_SPLIT}")
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
            if APP_CONFIG["LAUNCH_APP_FOR_EACH"] and ds and ds.count() > 0:
                logger.info(f"Обработка {ds.name} завершена.")

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
    main()
