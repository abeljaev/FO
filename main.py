import os
import glob
import json
import pandas as pd
import numpy as np  # Для работы с эмбеддингами
import fiftyone as fo
import fiftyone.zoo as foz  # Для моделей эмбеддингов
from sklearn.metrics.pairwise import (
    euclidean_distances,
)  # Для Farthest Point Sampling (если понадобится позже)

from config import (
    PATH_TO_SPLIT,
    PATH_TO_PREDICTIONS,
    SIZE_REQUIREMENTS,
    OUR_TO_MODEL_CLASSES,
    MODEL_MAPPING,
    CLASSES_GROUPS,
)

IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}

SKIP = {"safety", "no_safety", "chin_strap", "chin_strap_off", "glasses", "glasses_off"}
INCLUSION_THRESHOLD_GT_COVERED = 0.8


# --- Вспомогательные функции (get_group_for_label, map_classes, map_gt_class) ---
# (Оставляем как есть из предыдущего варианта)
def get_group_for_label(label: str) -> str:
    for group, labels in CLASSES_GROUPS.items():
        if label in labels:
            return group
    return label


def map_classes(pred_label):
    for target_label, all_labels in MODEL_MAPPING.items():
        if pred_label in all_labels:
            return target_label
    return pred_label


def map_gt_class(gt_label):
    if gt_label in OUR_TO_MODEL_CLASSES:
        return list(OUR_TO_MODEL_CLASSES[gt_label])[0]
    return gt_label


# --- Вспомогательные функции для геометрии (get_abs_bbox_from_normalized, calculate_area, calculate_intersection_area) ---
# (Оставляем как есть из предыдущего варианта)
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


# --- Кастомная оценка по вхождению (evaluate_by_inclusion) ---
# (Оставляем как есть из предыдущего варианта)
def evaluate_by_inclusion(
    dataset,
    gt_field="ground_truth",
    pred_field="predictions",
    gt_covered_threshold=INCLUSION_THRESHOLD_GT_COVERED,
):
    print(f"Running custom inclusion evaluation for dataset: {dataset.name}")
    view = dataset.view()

    for sample in view.iter_samples(autosave=True, progress=True):
        # Проверка наличия metadata.width и metadata.height
        if (
            sample.metadata is None
            or sample.metadata.width is None
            or sample.metadata.height is None
        ):
            # print(f"Warning: Sample {sample.filepath} is missing metadata (width/height). Attempting to compute.")
            try:
                sample.compute_metadata(overwrite=False)  # Пытаемся вычислить, если нет
                if (
                    sample.metadata is None or sample.metadata.width is None
                ):  # Проверяем снова
                    print(
                        f"Error: Could not compute metadata for {sample.filepath}. Skipping inclusion eval for this sample."
                    )
                    continue
            except Exception as e:
                print(
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
    print(f"Custom inclusion evaluation complete for {dataset.name}.")


# --- Функция для вычисления эмбеддингов патчей ---
def compute_and_save_patch_embeddings(
    dataset_or_view,
    patches_field="ground_truth",  # Для каких объектов считать: GT или Predictions
    model_name="clip-vit-base-patch32-torch",  # Модель для эмбеддингов
    embeddings_storage_field="patch_embeddings",  # Имя поля для сохранения в Detection
):
    """
    Вычисляет и сохраняет эмбеддинги для патчей (объектов) в датасете.
    """
    if not isinstance(dataset_or_view, (fo.Dataset, fo.DatasetView)):
        print(
            "Ошибка: Первый аргумент должен быть объектом fo.Dataset или fo.DatasetView."
        )
        return

    # Имя поля, куда будут сохранены эмбеддинги для каждого объекта
    # Добавим префикс от patches_field, чтобы было понятно, чьи это эмбеддинги
    final_embeddings_field = f"{patches_field}_{embeddings_storage_field}"

    # Проверяем, есть ли вообще поле с детекциями
    if not dataset_or_view.has_sample_field(patches_field):
        print(
            f"Поле '{patches_field}' не найдено в датасете/view. Пропуск вычисления эмбеддингов."
        )
        return

    # Создаем view только с сэмплами, где есть детекции в patches_field
    # и у этих детекций еще нет посчитанных эмбеддингов в target field
    # view_to_process = dataset_or_view.filter_labels(patches_field, fo.ViewField(final_embeddings_field).exists() == False)
    # Более простой способ - просто вычислить для всех, compute_embeddings может иметь опцию overwrite

    # Проверим, есть ли вообще объекты в patches_field
    # Собираем все метки из поля patches_field.detections.label
    # distinct_labels = dataset_or_view.distinct(f"{patches_field}.detections.label")
    # if not distinct_labels: # Если список меток пуст, значит объектов нет
    #     print(f"В датасете/view нет объектов в поле '{patches_field}' для вычисления эмбеддингов. Пропуск.")
    #     return

    # Проверка на наличие самих detections
    # Считаем количество сэмплов, у которых есть хотя бы одна детекция в patches_field
    # Это может быть медленно на больших датасетах, но fo.compute_embeddings справится с пустыми
    # count_samples_with_patches = dataset_or_view.count(fo.ViewField(f"{patches_field}.detections").length() > 0)
    # if count_samples_with_patches == 0:
    #     print(f"В датасете/view нет объектов в поле '{patches_field}' для вычисления эмбеддингов. Пропуск.")
    #     return

    print(
        f"Вычисление эмбеддингов для объектов из поля '{patches_field}' датасета '{dataset_or_view.name}'."
    )
    print(
        f"Модель: {model_name}. Результат будет сохранен в поле '{final_embeddings_field}' каждой детекции."
    )

    try:
        fo.compute_embeddings(
            dataset_or_view,
            model_name,
            embeddings_field=final_embeddings_field,
            patches_field=patches_field,
            # batch_size=10 # Можно настроить для управления памятью/скоростью
        )
        print(f"Эмбеддинги успешно вычислены и сохранены.")
        if isinstance(dataset_or_view, fo.Dataset):  # Если это датасет, а не view
            dataset_or_view.save()  # Сохраняем изменения в датасете
    except Exception as e:
        print(f"Ошибка при вычислении эмбеддингов: {e}")
        print(
            "Убедитесь, что модель существует и все зависимости установлены (torch, torchvision, transformers и т.д.)."
        )
        print(
            "Также проверьте, что в `patches_field` действительно есть объекты Detection."
        )


def load_class_dataset_from_csv(
    csv_file,
    predictions_dict,
    iou_dict={0.7: "IOU_0.7"},
    launch_app_on_completion=False,
    port=30082,
    compute_embeddings_for_gt=False,  # Новый параметр
    embeddings_model="clip-vit-base-patch32-torch",  # Модель для эмбеддингов
):
    base_name = os.path.basename(csv_file)
    class_name = os.path.splitext(base_name)[0]

    print(f"\n=== Обработка CSV: {csv_file} (класс: {class_name}) ===")

    dataset_name = f"{class_name}"
    if dataset_name in fo.list_datasets():
        print(f"Dataset {dataset_name} уже существует. Удаление и создание заново.")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"])

    required_cols = {
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "instance_label",
    }
    if not required_cols.issubset(df.columns):
        print(
            f"Предупреждение: CSV {csv_file} не содержит всех обязательных столбцов: {required_cols}. Пропускаем файл."
        )
        return None  # Возвращаем None, если файл не может быть обработан

    processed_data = {}
    for _, row in df.iterrows():
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            # print(f"Предупреждение: Файл изображения {image_path} не найден. Пропуск строки.")
            continue

        image_name = row["image_name"]

        # Проверка на NaN или некорректные значения для размеров изображения
        try:
            w, h = int(row["image_width"]), int(row["image_height"])
            if w <= 0 or h <= 0:
                # print(f"Предупреждение: Некорректные размеры изображения (w={w}, h={h}) для {image_name} в {csv_file}. Пропуск строки.")
                continue
        except ValueError:
            # print(f"Предупреждение: Не удалось преобразовать размеры изображения в int для {image_name} в {csv_file}. Пропуск строки.")
            continue

        label = row["instance_label"]
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

        gt_detection_data = {
            "label": map_gt_class(label),
            "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            "box_width_abs": float(x2 - x1),  # Сохраняем абсолютную ширину/высоту
            "box_height_abs": float(y2 - y1),
            "cvat_task": CVAT_LINK
            + f'/tasks/{row["task_id"]}/jobs/{int(row["job_id"])}?frame={row["image_id"]}',
        }
        processed_data[image_path]["gt_detections"].append(gt_detection_data)

    for image_path, data in processed_data.items():
        image_name = data["image_name"]
        w, h = data["size"]

        if image_name in predictions_dict:
            preds_for_img = predictions_dict[image_name]
            if isinstance(preds_for_img, list):
                for pred in preds_for_img:
                    if not all(k in pred for k in ["label", "score", "bbox"]):
                        continue
                    pred_label_mapped = map_classes(pred["label"])
                    group = get_group_for_label(pred_label_mapped)
                    if group in SIZE_REQUIREMENTS:
                        min_width, min_height = SIZE_REQUIREMENTS[group]
                        x1p, y1p, x2p, y2p_abs = pred[
                            "bbox"
                        ]  # Предполагаем абсолютные из JSON
                        width_p_abs = x2p_abs - x1p
                        height_p_abs = y2p_abs - y1p
                        if width_p_abs < min_width or height_p_abs < min_height:
                            continue

                    x1p_abs, y1p_abs, x2p_abs, y2p_abs = pred["bbox"]
                    pred_detection_data = {
                        "label": pred_label_mapped,
                        "bounding_box": [
                            x1p_abs / w,
                            y1p_abs / h,
                            (x2p_abs - x1p_abs) / w,
                            (y2p_abs - y1p_abs) / h,
                        ],
                        "confidence": pred["score"],
                        "box_width_abs": float(x2p_abs - x1p_abs),
                        "box_height_abs": float(y2p_abs - y1p_abs),
                    }
                    data["pred_detections"].append(pred_detection_data)

    samples_to_add = []
    for image_path, data in processed_data.items():
        gt_objects = [fo.Detection(**d) for d in data["gt_detections"]]
        pred_objects = [fo.Detection(**d) for d in data["pred_detections"]]
        sample = fo.Sample(
            filepath=image_path,
            ground_truth=fo.Detections(detections=gt_objects),
            predictions=fo.Detections(detections=pred_objects),
        )
        # FiftyOne автоматически вычислит metadata (width, height из файла), если их нет.
        # Но так как мы их берем из CSV, они должны быть корректными.
        # Если есть сомнения, можно добавить sample.compute_metadata(overwrite=False)
        samples_to_add.append(sample)

    if samples_to_add:
        dataset.add_samples(samples_to_add)
        print(f"Добавлено {len(samples_to_add)} сэмплов в датасет {dataset_name}.")
    else:
        print(f"Нет валидных данных для добавления в датасет {dataset_name}.")
        return dataset  # Возвращаем пустой датасет, если он был создан

    # --- Оценка или кастомная логика ---
    if class_name in SKIP:
        print(
            f"Класс {class_name} находится в SKIP. Запуск кастомной оценки по вхождению."
        )
        evaluate_by_inclusion(
            dataset, gt_covered_threshold=INCLUSION_THRESHOLD_GT_COVERED
        )
    else:
        print(f"Класс {class_name} НЕ в SKIP. Запуск стандартной оценки IoU.")
        for iou_thr, iou_tag in iou_dict.items():
            print(f"Оценка для IoU = {iou_thr} (ключ: {iou_tag})")
            # Проверим, есть ли вообще GT детекции, чтобы избежать ошибки
            if dataset.count(f"ground_truth.detections") > 0:
                try:
                    dataset.evaluate_detections(
                        "predictions",
                        gt_field="ground_truth",
                        eval_key=iou_tag,
                        method="coco",
                        iou=iou_thr,
                        compute_mAP=False,
                        progress=True,
                    )
                except Exception as e:
                    print(
                        f"Ошибка при оценке для IoU {iou_thr} для класса {class_name}: {e}"
                    )
            else:
                print(f"Пропуск оценки IoU для {class_name}: нет GT детекций.")

    # --- Вычисление эмбеддингов ---
    if compute_embeddings_for_gt:
        print(f"Вычисление эмбеддингов для GT объектов датасета {dataset_name}...")
        compute_and_save_patch_embeddings(
            dataset,
            patches_field="ground_truth",  # Считаем для GT
            model_name=embeddings_model,
            embeddings_storage_field="clip_embeddings",  # Пример имени поля
        )
        # Можно также посчитать для predictions, если нужно
        # compute_and_save_patch_embeddings(
        #     dataset,
        #     patches_field="predictions",
        #     model_name=embeddings_model,
        #     embeddings_storage_field="clip_embeddings"
        # )

    print(f"\nДатасет {dataset_name} обработан и сохранен.")
    session = None
    if launch_app_on_completion:
        print(f"Запуск FiftyOne App для датасета {dataset_name} на порту {port}...")
        try:
            session = fo.launch_app(dataset, address="0.0.0.0", port=port, auto=False)
            print(
                f"FiftyOne App должен быть доступен по адресу: http://<ваш_ip>:{port} (или localhost:{port})"
            )
        except Exception as e:
            print(f"Не удалось запустить FiftyOne App: {e}")
            print(
                "Возможно, порт уже занят или есть другая проблема с запуском сервера."
            )

    return dataset


def main():
    LAUNCH_APP_FOR_EACH = False  # Открывать браузер для каждого датасета?
    COMPUTE_GT_EMBEDDINGS = True  # Вычислять эмбеддинги для GT патчей?
    EMBEDDINGS_MODEL_NAME = "clip-vit-base-patch32-torch"  # Модель для эмбеддингов

    # Убедитесь, что модель существует и зависимости установлены.
    # fo.zoo.models.list_downloadable_models(tags="embedding") для списка доступных.

    start_port = (
        30082  # Начальный порт для FiftyOne App, если LAUNCH_APP_FOR_EACH = True
    )

    try:
        with open(PATH_TO_PREDICTIONS, "r") as f:
            predictions_dict = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл с предсказаниями не найден: {PATH_TO_PREDICTIONS}")
        return
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON из файла: {PATH_TO_PREDICTIONS}")
        return

    csv_files = sorted(glob.glob(os.path.join(PATH_TO_SPLIT, "*.csv")))
    if not csv_files:
        print(f"CSV файлы не найдены в {PATH_TO_SPLIT}")
        return

    loaded_datasets_info = []  # Будем хранить имя и был ли запущен app

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
        if dataset:
            loaded_datasets_info.append(
                {
                    "name": dataset.name,
                    "app_launched": LAUNCH_APP_FOR_EACH,  # Был ли сделан вызов launch_app
                    "port": port_for_this_dataset if LAUNCH_APP_FOR_EACH else None,
                }
            )
            if LAUNCH_APP_FOR_EACH and dataset:  # Если запускали приложение
                print(f"Обработка датасета {dataset.name} завершена.")
                # input("Нажмите Enter для обработки следующего файла (если сессия не блокирует)...")
                print("-" * 30)

    print("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_info:
        print("Следующие датасеты были созданы/обновлены в FiftyOne:")
        for info in loaded_datasets_info:
            msg = f"- {info['name']}"
            if COMPUTE_GT_EMBEDDINGS:
                msg += " (с эмбеддингами для GT)"
            if info["app_launched"]:
                msg += f" (приложение запущено на порту {info['port']})"
            print(msg)

        print(
            "\nЧтобы просмотреть датасет (если приложение не было запущено автоматически):"
        )
        print("import fiftyone as fo")
        if loaded_datasets_info:  # Проверка, что список не пуст
            example_name = loaded_datasets_info[0]["name"]
            print(f"dataset = fo.load_dataset('{example_name}')")
            print(
                f"session = fo.launch_app(dataset, port={start_port}) # Укажите нужный порт"
            )
            print("session.wait()")
        print("\nВ FiftyOne App:")
        print(
            "  - Для классов из SKIP: смотрите кастомные поля в детекциях (например, 'max_pred_inclusion_in_gt')."
        )
        print(
            "  - Если эмбеддинги были посчитаны (для 'ground_truth_patch_embeddings' или аналогичного поля):"
        )
        print("    - Откройте панель 'Embeddings'.")
        print("    - Выберите в 'Label field' -> 'ground_truth.detections'.")
        print(
            "    - В 'Field with embeddings' выберите 'ground_truth_clip_embeddings' (или как вы назвали поле)."
        )
        print("    - Нажмите 'Compute visualization' (например, UMAP).")
    else:
        print("Не было создано ни одного датасета.")


if __name__ == "__main__":
    main()
