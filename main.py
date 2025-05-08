import os
import glob
import json
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from loguru import logger  # Импортируем loguru

# Настройка Loguru: выводим только INFO и выше в консоль, можно добавить файл
logger.remove()  # Удаляем стандартный обработчик
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
# logger.add("file_{time}.log", level="DEBUG") # Опционально: запись в файл

from config import (
    PATH_TO_SPLIT,
    PATH_TO_PREDICTIONS,
    SIZE_REQUIREMENTS,
    OUR_TO_MODEL_CLASSES,  # {our_class: {model_class1, model_class2}}
    MODEL_MAPPING,  # {target_canonical_class: {model_class_alias1, ...}}
    CLASSES_GROUPS,
)

IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}

SKIP = {  # Классы для кастомной оценки по вхождению
    "safety",
    "no_safety",
    "chin_strap",
    "chin_strap_off",
    "glasses",
    "glasses_off",
}
INCLUSION_THRESHOLD_GT_COVERED = 0.8


# --- Вспомогательные функции (get_group_for_label, map_classes, map_gt_class) ---
def get_group_for_label(label: str) -> str:
    for group, labels in CLASSES_GROUPS.items():
        if label in labels:
            return group
    return label


def map_classes_predictions(pred_label_from_model: str) -> str:
    """Маппинг метки предсказания модели к канонической метке."""
    for target_label, source_model_labels in MODEL_MAPPING.items():
        if pred_label_from_model in source_model_labels:
            return target_label
    return pred_label_from_model  # Если нет маппинга, возвращаем как есть


def map_gt_label_to_model_label_set(gt_label_our: str) -> set:
    """
    Преобразование нашей GT метки в НАБОР соответствующих меток модели.
    Это важно, так как один наш класс может соответствовать нескольким классам модели.
    """
    return OUR_TO_MODEL_CLASSES.get(gt_label_our, {gt_label_our})


# --- Вспомогательные функции для геометрии ---
# (Оставляем как есть)
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


# --- Кастомная оценка по вхождению ---
# (Оставляем как есть, но логирование через logger)
def evaluate_by_inclusion(
    dataset,
    gt_field="ground_truth",
    pred_field="predictions",
    gt_covered_threshold=INCLUSION_THRESHOLD_GT_COVERED,
):
    logger.info(f"Running custom inclusion evaluation for dataset: {dataset.name}")
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
                        f"Could not compute metadata for {sample.filepath}. Skipping inclusion eval for this sample."
                    )
                    continue
            except Exception as e:
                logger.error(
                    f"Error computing metadata for {sample.filepath}: {e}. Skipping inclusion eval for this sample."
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
                "det": gt_det,
                "abs_bbox": get_abs_bbox_from_normalized(
                    gt_det.bounding_box, img_width, img_height
                ),
            }
            for gt_det in gt_detections
        ]
        abs_preds = [
            {
                "det": pred_det,
                "abs_bbox": get_abs_bbox_from_normalized(
                    pred_det.bounding_box, img_width, img_height
                ),
            }
            for pred_det in pred_detections
        ]

        for gt_item in abs_gts:
            gt_det = gt_item["det"]
            gt_abs_bbox = gt_item["abs_bbox"]
            gt_area = calculate_area(gt_abs_bbox)
            gt_det["max_pred_inclusion_in_gt"] = 0.0
            gt_det["gt_covered_by_inclusion"] = False
            if gt_area > 0 and abs_preds:
                for pred_item in abs_preds:
                    pred_abs_bbox = pred_item["abs_bbox"]
                    intersection_area = calculate_intersection_area(
                        gt_abs_bbox, pred_abs_bbox
                    )
                    inclusion_in_gt = intersection_area / gt_area if gt_area > 0 else 0
                    if inclusion_in_gt > gt_det["max_pred_inclusion_in_gt"]:
                        gt_det["max_pred_inclusion_in_gt"] = inclusion_in_gt
                if gt_det["max_pred_inclusion_in_gt"] >= gt_covered_threshold:
                    gt_det["gt_covered_by_inclusion"] = True

        for pred_item in abs_preds:
            pred_det = pred_item["det"]
            pred_abs_bbox = pred_item["abs_bbox"]
            pred_area = calculate_area(pred_abs_bbox)
            pred_det["max_gt_inclusion_in_pred"] = 0.0
            pred_det["max_gt_coverage_by_pred"] = 0.0
            if pred_area > 0 and abs_gts:
                for gt_item in abs_gts:
                    gt_abs_bbox = gt_item["abs_bbox"]
                    gt_area = calculate_area(gt_abs_bbox)
                    intersection_area = calculate_intersection_area(
                        gt_abs_bbox, pred_abs_bbox
                    )
                    inclusion_in_pred = (
                        intersection_area / pred_area if pred_area > 0 else 0
                    )
                    if inclusion_in_pred > pred_det["max_gt_inclusion_in_pred"]:
                        pred_det["max_gt_inclusion_in_pred"] = inclusion_in_pred
                    coverage_of_gt = intersection_area / gt_area if gt_area > 0 else 0
                    if coverage_of_gt > pred_det["max_gt_coverage_by_pred"]:
                        pred_det["max_gt_coverage_by_pred"] = coverage_of_gt
    logger.info(f"Custom inclusion evaluation complete for {dataset.name}.")


# --- Функция для вычисления эмбеддингов патчей ---
# (Оставляем как есть, но логирование через logger)
def compute_and_save_patch_embeddings(
    dataset_or_view,
    patches_field="ground_truth",
    model_name="clip-vit-base-patch32-torch",
    embeddings_storage_field="patch_embeddings",
):
    if not isinstance(dataset_or_view, (fo.Dataset, fo.DatasetView)):
        logger.error(
            "Первый аргумент должен быть объектом fo.Dataset или fo.DatasetView."
        )
        return

    final_embeddings_field = f"{patches_field}_{embeddings_storage_field}"
    if not dataset_or_view.has_sample_field(patches_field):
        logger.warning(
            f"Поле '{patches_field}' не найдено в датасете/view {dataset_or_view.name}. Пропуск вычисления эмбеддингов."
        )
        return

    logger.info(
        f"Вычисление эмбеддингов для объектов из поля '{patches_field}' датасета '{dataset_or_view.name}'."
    )
    logger.info(
        f"Модель: {model_name}. Результат будет сохранен в поле '{final_embeddings_field}' каждой детекции."
    )

    try:
        # Проверим, есть ли вообще объекты в patches_field, чтобы не вызывать compute_embeddings зря
        # Считаем количество детекций в этом поле по всему датасету/view
        # Этот способ может быть не самым быстрым для очень больших датасетов, но надежен
        total_detections_in_field = 0
        for sample in dataset_or_view.select_fields(f"{patches_field}.detections"):
            if sample[patches_field] and sample[patches_field].detections:
                total_detections_in_field += len(sample[patches_field].detections)

        if total_detections_in_field == 0:
            logger.info(
                f"В поле '{patches_field}' датасета '{dataset_or_view.name}' нет объектов. Пропуск вычисления эмбеддингов."
            )
            return

        fo.compute_embeddings(
            dataset_or_view,
            model_name,
            embeddings_field=final_embeddings_field,
            patches_field=patches_field,
        )
        logger.info(
            f"Эмбеддинги успешно вычислены и сохранены для {dataset_or_view.name}."
        )
        if isinstance(dataset_or_view, fo.Dataset):
            dataset_or_view.save()
    except Exception as e:
        logger.error(
            f"Ошибка при вычислении эмбеддингов для {dataset_or_view.name}: {e}"
        )
        logger.error(
            "Убедитесь, что модель существует и все зависимости установлены (torch, torchvision, transformers и т.д.)."
        )


def load_class_dataset_from_csv(
    csv_file,
    predictions_dict,  # Словарь всех предсказаний {image_name: [preds]}
    iou_dict={0.7: "IOU_0.7"},
    launch_app_on_completion=False,
    port=30082,
    compute_embeddings_for_gt=False,
    embeddings_model="clip-vit-base-patch32-torch",
):
    base_name = os.path.basename(csv_file)
    # `current_class_name_our` - это имя нашего класса, из имени CSV файла
    current_class_name_our = os.path.splitext(base_name)[0]

    logger.info(
        f"=== Обработка CSV: {csv_file} (Наш класс: {current_class_name_our}) ==="
    )

    dataset_name = f"{current_class_name_our}"  # Имя датасета = имя нашего класса
    if dataset_name in fo.list_datasets():
        logger.info(
            f"Dataset {dataset_name} уже существует. Удаление и создание заново."
        )
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

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

    # Получаем набор меток модели, которые соответствуют НАШЕМУ текущему классу из CSV
    # Это будет использоваться для фильтрации GT и Предсказаний
    # `gt_label_our` из CSV должен быть current_class_name_our (или мы можем проверить это)
    # А `map_gt_label_to_model_label_set` преобразует `current_class_name_our` в целевые метки модели.
    target_model_labels_for_this_csv_class = map_gt_label_to_model_label_set(
        current_class_name_our
    )
    if not target_model_labels_for_this_csv_class:  # Если маппинг не дал результатов
        logger.warning(
            f"Для нашего класса '{current_class_name_our}' не найдено соответствий в классах модели (OUR_TO_MODEL_CLASSES). Предсказания могут быть не отфильтрованы корректно."
        )
        # В этом случае, возможно, стоит использовать current_class_name_our как есть, или пропускать?
        # Пока оставим так, предсказания будут фильтроваться по каноническим классам после map_classes_predictions.

    processed_data = (
        {}
    )  # {image_path: {"gt_detections": [], "pred_detections": [], "size": (w,h), "image_name": name}}

    for _, row in df.iterrows():
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            logger.warning(
                f"Файл изображения {image_path} из {csv_file} не найден. Пропуск строки."
            )
            continue

        image_name = row["image_name"]
        gt_label_our_from_row = row["instance_label"]

        # Проверяем, что метка GT в строке CSV соответствует классу, для которого этот CSV предназначен
        if gt_label_our_from_row != current_class_name_our:
            logger.debug(
                f"Пропуск GT метки '{gt_label_our_from_row}' в CSV для класса '{current_class_name_our}' (файл {csv_file})."
            )
            continue

        try:
            w, h = int(row["image_width"]), int(row["image_height"])
            if w <= 0 or h <= 0:
                logger.warning(
                    f"Некорректные размеры изображения (w={w}, h={h}) для {image_name} в {csv_file}. Пропуск строки."
                )
                continue
        except ValueError:
            logger.warning(
                f"Не удалось преобразовать размеры изображения в int для {image_name} в {csv_file}. Пропуск строки."
            )
            continue

        x1, y1, x2, y2 = (
            row["bbox_x_tl"],
            row["bbox_y_tl"],
            row["bbox_x_br"],
            row["bbox_y_br"],
        )

        if image_path not in processed_data:
            processed_data[image_path] = {
                "gt_detections": [],
                "pred_detections": [],
                "size": (w, h),
                "image_name": image_name,
            }

        # GT метки для FiftyOne должны быть уже смаплены к каноническим классам модели,
        # которые мы ожидаем в предсказаниях, для корректной работы evaluate_detections.
        # Но так как `target_model_labels_for_this_csv_class` это set, возьмем первый элемент
        # или, если их несколько, то это требует более сложной логики, если evaluate_detections
        # не умеет работать с GT типа "или класс А или класс Б".
        # Обычно GT имеет одну метку.
        # Используем первый элемент из target_model_labels_for_this_csv_class для GT.
        # Если current_class_name_our не мапится, используем его как есть (уже обработано в map_gt_label_to_model_label_set).
        display_gt_label = (
            list(target_model_labels_for_this_csv_class)[0]
            if target_model_labels_for_this_csv_class
            else current_class_name_our
        )

        gt_detection_data = {
            "label": display_gt_label,  # Используем смапленную метку для GT
            "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            "box_width_abs": float(x2 - x1),
            "box_height_abs": float(y2 - y1),
            "cvat_task": CVAT_LINK
            + f'/tasks/{row["task_id"]}/jobs/{int(row["job_id"])}?frame={row["image_id"]}',
        }
        processed_data[image_path]["gt_detections"].append(gt_detection_data)

    # Добавляем предсказания, отфильтрованные для текущего класса
    for image_path, data in processed_data.items():
        image_name = data["image_name"]
        w, h = data["size"]

        if image_name in predictions_dict:
            preds_for_img = predictions_dict[image_name]
            if isinstance(preds_for_img, list):
                for pred_raw in preds_for_img:
                    if not all(k in pred_raw for k in ["label", "score", "bbox"]):
                        continue

                    # Шаг 1: Маппинг метки предсказания из модели к нашему каноническому виду
                    pred_label_canonical = map_classes_predictions(pred_raw["label"])

                    # Шаг 2: Фильтрация. Предсказание добавляется, если его каноническая метка
                    #          находится в `target_model_labels_for_this_csv_class`.
                    #          Это гарантирует, что мы добавляем только те предсказания,
                    #          которые релевантны для ГТ данного CSV файла.
                    if (
                        pred_label_canonical
                        not in target_model_labels_for_this_csv_class
                    ):
                        continue  # Этот предсказанный класс не соответствует текущему CSV

                    # Фильтрация предсказаний по размерам (применяется к каноническому классу ПРЕДСКАЗАНИЯ)
                    group = get_group_for_label(pred_label_canonical)
                    if group in SIZE_REQUIREMENTS:
                        min_width, min_height = SIZE_REQUIREMENTS[group]
                        x1p_abs_raw, y1p_abs_raw, x2p_abs_raw, y2p_abs_raw = pred_raw[
                            "bbox"
                        ]
                        width_p_abs = x2p_abs_raw - x1p_abs_raw
                        height_p_abs = y2p_abs_raw - y1p_abs_raw
                        if width_p_abs < min_width or height_p_abs < min_height:
                            continue

                    x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred_raw["bbox"]
                    pred_detection_data = {
                        "label": pred_label_canonical,  # Используем каноническую метку
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
                    data["pred_detections"].append(pred_detection_data)

    samples_to_add = []
    for image_path, data in processed_data.items():
        # Только если есть GT объекты для этого класса, добавляем сэмпл
        if not data["gt_detections"]:
            continue

        gt_objects = [fo.Detection(**d) for d in data["gt_detections"]]
        pred_objects = [fo.Detection(**d) for d in data["pred_detections"]]
        sample = fo.Sample(
            filepath=image_path,
            ground_truth=fo.Detections(detections=gt_objects),
            predictions=fo.Detections(
                detections=pred_objects
            ),  # Это поле используется для evaluate_detections
        )
        samples_to_add.append(sample)

    if samples_to_add:
        dataset.add_samples(samples_to_add)
        logger.info(
            f"Добавлено {len(samples_to_add)} сэмплов в датасет {dataset_name}."
        )
    else:
        logger.warning(
            f"Нет валидных данных (с GT для класса {current_class_name_our}) для добавления в датасет {dataset_name}."
        )
        # fo.delete_dataset(dataset_name) # Можно удалить пустой датасет, если он был создан
        return dataset  # Возвращаем датасет (возможно, пустой, но созданный)

    # --- Оценка или кастомная логика ---
    if current_class_name_our in SKIP:
        logger.info(
            f"Класс {current_class_name_our} находится в SKIP. Запуск кастомной оценки по вхождению."
        )
        evaluate_by_inclusion(
            dataset, gt_covered_threshold=INCLUSION_THRESHOLD_GT_COVERED
        )
    else:
        logger.info(
            f"Класс {current_class_name_our} НЕ в SKIP. Запуск стандартной оценки IoU."
        )
        # Убедимся, что в GT есть детекции с нужной меткой (одной из target_model_labels_for_this_csv_class)
        # evaluate_detections ожидает, что GT и pred метки совпадают.
        # `display_gt_label` - это та метка, которую мы присвоили GT детекциям.
        # Если target_model_labels_for_this_csv_class содержит несколько меток,
        # то evaluate_detections нужно вызывать для каждой из них или настроить classes параметр.
        # Для простоты, если `target_model_labels_for_this_csv_class` > 1, мы можем оценить по `display_gt_label`
        # или пропустить, если это нежелательно.

        # `classes` параметр в evaluate_detections позволяет указать, для каких классов считать.
        # Мы хотим считать только для `display_gt_label` в этом датасете.
        eval_classes_list = [display_gt_label]

        if (
            dataset.count(f"ground_truth.detections") > 0
        ):  # Проверяем наличие GT в целом
            for iou_thr, iou_tag in iou_dict.items():
                logger.info(
                    f"Оценка для IoU = {iou_thr} (ключ: {iou_tag}), классы: {eval_classes_list}"
                )
                try:
                    dataset.evaluate_detections(
                        "predictions",  # Имя поля с предсказаниями
                        gt_field="ground_truth",
                        eval_key=iou_tag,
                        method="coco",
                        iou=iou_thr,
                        compute_mAP=False,  # mAP не нужен для одного класса
                        classes=eval_classes_list,  # Явно указываем класс для оценки
                        progress=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка при оценке для IoU {iou_thr} для класса {current_class_name_our} (метка модели {display_gt_label}): {e}"
                    )
        else:
            logger.warning(
                f"Пропуск оценки IoU для {current_class_name_our}: нет GT детекций в датасете."
            )

    # --- Вычисление эмбеддингов ---
    if compute_embeddings_for_gt:
        logger.info(
            f"Вычисление эмбеддингов для GT объектов датасета {dataset_name}..."
        )
        compute_and_save_patch_embeddings(
            dataset,
            patches_field="ground_truth",
            model_name=embeddings_model,
            embeddings_storage_field="clip_embeddings",
        )

    logger.info(f"Датасет {dataset_name} обработан и сохранен.")
    session = None
    if launch_app_on_completion:
        logger.info(
            f"Запуск FiftyOne App для датасета {dataset_name} на порту {port}..."
        )
        try:
            session = fo.launch_app(dataset, address="0.0.0.0", port=port, auto=False)
            logger.info(
                f"FiftyOne App должен быть доступен по адресу: http://<ваш_ip>:{port} (или localhost:{port})"
            )
        except Exception as e:
            logger.error(f"Не удалось запустить FiftyOne App для {dataset_name}: {e}")

    return dataset


def main():
    LAUNCH_APP_FOR_EACH = False
    COMPUTE_GT_EMBEDDINGS = True
    EMBEDDINGS_MODEL_NAME = "clip-vit-base-patch32-torch"

    start_port = 30082

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            predictions_dict = json.load(f)
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

    loaded_datasets_info = []

    for i, csv_file in enumerate(csv_files):
        port_for_this_dataset = start_port + i if LAUNCH_APP_FOR_EACH else start_port

        dataset = load_class_dataset_from_csv(
            csv_file=csv_file,
            predictions_dict=predictions_dict,
            iou_dict=IOU_DICT,
            launch_app_on_completion=LAUNCH_APP_FOR_EACH,
            port=port_for_this_dataset,
            compute_embeddings_for_gt=COMPUTE_GT_EMBEDDINGS,
            embeddings_model=EMBEDDINGS_MODEL_NAME,
        )
        if (
            dataset
        ):  # Если датасет был успешно создан (даже если пустой, но был return dataset)
            loaded_datasets_info.append(
                {
                    "name": dataset.name,
                    "app_launched": LAUNCH_APP_FOR_EACH
                    and dataset.count() > 0,  # Запускаем только если не пустой
                    "port": (
                        port_for_this_dataset
                        if LAUNCH_APP_FOR_EACH and dataset.count() > 0
                        else None
                    ),
                    "has_embeddings": COMPUTE_GT_EMBEDDINGS
                    and dataset.has_field(f"ground_truth_clip_embeddings"),
                }
            )
            if LAUNCH_APP_FOR_EACH and dataset and dataset.count() > 0:
                logger.info(f"Обработка датасета {dataset.name} завершена.")
                # input("Нажмите Enter для обработки следующего файла...") # Для пошагового просмотра
                logger.info("-" * 30)

    logger.info("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_info:
        logger.info("Следующие датасеты были созданы/обновлены в FiftyOne:")
        for info in loaded_datasets_info:
            msg = f"- {info['name']}"
            if info["has_embeddings"]:
                msg += " (с эмбеддингами для GT)"
            if info["app_launched"]:
                msg += f" (приложение запущено на порту {info['port']})"
            logger.info(msg)

        logger.info(
            "\nЧтобы просмотреть датасет (если приложение не было запущено автоматически):"
        )
        logger.info("import fiftyone as fo")
        if loaded_datasets_info:
            example_name = loaded_datasets_info[0]["name"]
            logger.info(f"dataset = fo.load_dataset('{example_name}')")
            logger.info(f"session = fo.launch_app(dataset, port={start_port})")
            logger.info("session.wait()")
    else:
        logger.warning("Не было создано ни одного датасета.")


if __name__ == "__main__":
    # fo.config.show_progress_bars = False # Можно отключить прогресс-бары fiftyone, если логи loguru достаточно
    main()
