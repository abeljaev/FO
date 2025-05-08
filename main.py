import os
import glob
import json
import pandas as pd
import fiftyone as fo
from config import (
    PATH_TO_SPLIT,
    PATH_TO_PREDICTIONS,
    SIZE_REQUIREMENTS,
    OUR_TO_MODEL_CLASSES,
    MODEL_MAPPING,
    CLASSES_GROUPS,
    CVAT_LINK,
)

IOU_DICT = {0.4: "eval_IOU_04", 0.7: "eval_IOU_07"}

# SKIP теперь означает использование кастомной метрики вхождения
SKIP = {"safety", "no_safety", "chin_strap", "chin_strap_off", "glasses", "glasses_off"}

# Порог для кастомной метрики вхождения (GT должен быть покрыт хотя бы на этот процент)
INCLUSION_THRESHOLD_GT_COVERED = 0.8  # Например, 80% GT должно быть внутри предикта


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


# --- Вспомогательные функции для геометрии ---
def get_abs_bbox_from_normalized(norm_bbox, img_width, img_height):
    """Преобразует нормализованный bbox [x, y, w, h] в абсолютный [x1, y1, x2, y2]."""
    x, y, w, h = norm_bbox
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h) * img_height
    return [x1, y1, x2, y2]


def calculate_area(bbox_abs):
    """bbox_abs: [x1, y1, x2, y2]"""
    if None in bbox_abs or bbox_abs[2] < bbox_abs[0] or bbox_abs[3] < bbox_abs[1]:
        return 0.0
    return (bbox_abs[2] - bbox_abs[0]) * (bbox_abs[3] - bbox_abs[1])


def calculate_intersection_area(bbox1_abs, bbox2_abs):
    """bbox_abs: [x1, y1, x2, y2]"""
    x_left = max(bbox1_abs[0], bbox2_abs[0])
    y_top = max(bbox1_abs[1], bbox2_abs[1])
    x_right = min(bbox1_abs[2], bbox2_abs[2])
    y_bottom = min(bbox1_abs[3], bbox2_abs[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


# --- Кастомная оценка по вхождению ---
def evaluate_by_inclusion(
    dataset,
    gt_field="ground_truth",
    pred_field="predictions",
    gt_covered_threshold=INCLUSION_THRESHOLD_GT_COVERED,
):
    """
    Оценивает детекции на основе коэффициента вхождения.
    Добавляет поля к GT:
        - 'max_pred_inclusion_in_gt': max(Area(Intersection(GT, Pred)) / Area(GT))
        - 'gt_covered_by_inclusion': True, если 'max_pred_inclusion_in_gt' >= gt_covered_threshold
    Добавляет поля к Pred:
        - 'max_gt_inclusion_in_pred': max(Area(Intersection(GT, Pred)) / Area(Pred))
        - 'pred_contains_gt_significantly': True, если есть GT, для которого Area(I(GT,P))/Area(GT) высок
                                            (пока просто сохраним максимальный коэффициент)
    """
    print(f"Running custom inclusion evaluation for dataset: {dataset.name}")
    view = dataset.view()  # Process all samples

    for sample in view.iter_samples(autosave=True, progress=True):
        img_width, img_height = sample.metadata.width, sample.metadata.height
        if not img_width or not img_height:
            print(
                f"Skipping sample {sample.filepath} due to missing metadata (width/height)"
            )
            continue

        gt_detections = sample[gt_field].detections if sample[gt_field] else []
        pred_detections = sample[pred_field].detections if sample[pred_field] else []

        # Абсолютные координаты для всех GT и Pred заранее
        abs_gts = []
        for gt_det in gt_detections:
            abs_gts.append(
                {
                    "det": gt_det,
                    "abs_bbox": get_abs_bbox_from_normalized(
                        gt_det.bounding_box, img_width, img_height
                    ),
                }
            )

        abs_preds = []
        for pred_det in pred_detections:
            abs_preds.append(
                {
                    "det": pred_det,
                    "abs_bbox": get_abs_bbox_from_normalized(
                        pred_det.bounding_box, img_width, img_height
                    ),
                }
            )

        # Оценка для каждого GT
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
            # gt_det.save() # autosave=True в iter_samples должен это делать

        # Оценка для каждого Pred
        for pred_item in abs_preds:
            pred_det = pred_item["det"]
            pred_abs_bbox = pred_item["abs_bbox"]
            pred_area = calculate_area(pred_abs_bbox)
            pred_det["max_gt_inclusion_in_pred"] = 0.0  # Area(I(GT,P)) / Area(P)
            pred_det["max_gt_coverage_by_pred"] = 0.0  # Area(I(GT,P)) / Area(GT)

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
            # pred_det.save() # autosave=True

    print(f"Custom inclusion evaluation complete for {dataset.name}.")


def load_class_dataset_from_csv(
    csv_file,
    predictions_dict,
    iou_dict={0.7: "IOU_0.7"},  # Это останется для не-SKIP классов
    launch_app_on_completion=False,  # Новый параметр
    port=30082,  # Порт для запуска, если нужно
):
    base_name = os.path.basename(csv_file)
    class_name = os.path.splitext(base_name)[0]

    print(f"\n=== Обработка CSV: {csv_file} (класс: {class_name}) ===")

    dataset_name = f"{class_name}"  # Можно добавить префикс/суффикс для ясности
    if dataset_name in fo.list_datasets():
        print(f"Dataset {dataset_name} уже существует. Удаление и создание заново.")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True  # Важно для сохранения датасета

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
        raise ValueError(f"CSV {csv_file} должен содержать столбцы: {required_cols}")

    # Собираем GT и Preds в структуры, удобные для FiftyOne
    # {image_path: {"gt": [...], "pred": [...], "size": (w,h), "image_name": ...}}
    processed_data = {}

    for _, row in df.iterrows():
        image_path = row["image_path"]
        if not os.path.exists(image_path):  # Проверка существования файла GT
            # print(f"Предупреждение: Файл изображения {image_path} не найден. Пропуск.")
            continue

        image_name = row["image_name"]
        w, h = int(row["image_width"]), int(
            row["image_height"]
        )  # Убедимся, что это int
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
                "pred_detections": [],  # Заполним позже
                "size": (w, h),
                "image_name": image_name,
            }

        # Добавляем GT детекцию
        gt_detection_data = {
            "label": map_gt_class(label),
            "bounding_box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
            "abs_coords": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],  # Сохраняем абсолютные для кастомной метрики
            "box_width": x2 - x1,
            "box_height": y2 - y1,
            "cvat_task": CVAT_LINK
            + f'/tasks/{row["task_id"]}/jobs/{int(row["job_id"])}?frame={row["image_id"]}',
        }
        processed_data[image_path]["gt_detections"].append(gt_detection_data)

    # Добавляем предсказания
    for image_path, data in processed_data.items():
        image_name = data["image_name"]
        w, h = data["size"]

        if image_name in predictions_dict:
            preds_for_img = predictions_dict[image_name]
            # model_labels_for_class = OUR_TO_MODEL_CLASSES.get(class_name, {class_name}) # Для фильтрации по основному классу CSV

            if isinstance(preds_for_img, list):
                for pred in preds_for_img:
                    if not all(k in pred for k in ["label", "score", "bbox"]):
                        continue

                    pred_label_mapped = map_classes(pred["label"])
                    # Если мы хотим строгую привязку к классу CSV:
                    # if pred_label_mapped not in model_labels_for_class and class_name not in SKIP: # Для SKIP классов можем хотеть все предсказания
                    #     continue

                    # Фильтрация предсказаний по размерам (применяется к классу ПРЕДСКАЗАНИЯ)
                    group = get_group_for_label(
                        pred_label_mapped
                    )  # Используем смапленный лейбл предсказания
                    if group in SIZE_REQUIREMENTS:
                        min_width, min_height = SIZE_REQUIREMENTS[group]
                        x1p, y1p, x2p, y2p = pred["bbox"]
                        width_p = x2p - x1p
                        height_p = y2p - y1p
                        if width_p < min_width or height_p < min_height:
                            continue

                    x1p, y1p, x2p, y2p = pred["bbox"]
                    pred_detection_data = {
                        "label": pred_label_mapped,
                        "bounding_box": [
                            x1p / w,
                            y1p / h,
                            (x2p - x1p) / w,
                            (y2p - y1p) / h,
                        ],
                        "abs_coords": [
                            float(x1p),
                            float(y1p),
                            float(x2p),
                            float(y2p),
                        ],  # Сохраняем абсолютные
                        "confidence": pred["score"],
                        "box_width": x2p - x1p,
                        "box_height": y2p - y1p,
                    }
                    data["pred_detections"].append(pred_detection_data)

    # Создаем сэмплы FiftyOne
    samples_to_add = []
    for image_path, data in processed_data.items():
        gt_objects = [fo.Detection(**d) for d in data["gt_detections"]]
        pred_objects = [fo.Detection(**d) for d in data["pred_detections"]]

        sample = fo.Sample(
            filepath=image_path,
            ground_truth=fo.Detections(detections=gt_objects),
            predictions=fo.Detections(detections=pred_objects),
        )
        # Добавляем metadata, если его нет (FiftyOne обычно делает это автоматически при добавлении)
        # sample.compute_metadata() # Это нужно если width/height не были в CSV и их надо извлечь из файла
        samples_to_add.append(sample)

    if samples_to_add:
        dataset.add_samples(samples_to_add)
        print(f"Добавлено {len(samples_to_add)} сэмплов в датасет {dataset_name}.")
    else:
        print(f"Нет данных для добавления в датасет {dataset_name}.")
        # fo.delete_dataset(dataset_name) # Можно удалить пустой датасет
        return None  # или dataset, если хотим сохранить пустой

    # --- Оценка ---
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
            try:
                dataset.evaluate_detections(
                    "predictions",
                    gt_field="ground_truth",
                    eval_key=iou_tag,
                    method="coco",  # стандартный метод
                    iou=iou_thr,
                    compute_mAP=False,  # Пока mAP не считаем для отдельных классов
                    progress=True,
                )
            except Exception as e:
                print(
                    f"Ошибка при оценке для IoU {iou_thr} для класса {class_name}: {e}"
                )
                print(
                    "Возможно, нет GT или предсказаний для этого класса, или они не пересекаются."
                )

    print(f"\nДатасет {dataset_name} обработан и сохранен.")

    session = None
    if launch_app_on_completion:
        print(f"Запуск FiftyOne App для датасета {dataset_name} на порту {port}...")
        session = fo.launch_app(
            dataset, address="0.0.0.0", port=port, auto=False
        )  # auto=False чтобы не открывать сразу
        print(f"FiftyOne App доступен по адресу: http://<ваш_ip>:{port}")

    return dataset  # Возвращаем датасет, вдруг он понадобится в main


def main():
    LAUNCH_APP_INTERACTIVELY = (
        False  # True, если хотите открывать браузер для каждого датасета
    )
    # Или можно сделать более сложную логику:
    # LAUNCH_APP_FOR_LAST_ONLY = True
    # LAUNCH_APP_MANUALLY_LATER = True (по умолчанию, ничего не запускаем из скрипта)

    # Установим глобальный порт для сессий, если нужно, чтобы они не конфликтовали
    # fo.config.default_app_port = 30082 # Если хотите фиксированный порт по умолчанию

    current_port = 30082  # Начальный порт

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

    loaded_datasets_names = []

    for i, csv_file in enumerate(csv_files):
        # Логика для порта: или инкрементировать, или использовать один и тот же,
        # но тогда предыдущая сессия закроется при запуске новой на том же порту.
        # Если LAUNCH_APP_INTERACTIVELY = False, порт не так важен здесь.
        port_for_this_dataset = current_port  # + i # если хотим разные порты

        dataset = load_class_dataset_from_csv(
            csv_file=csv_file,
            predictions_dict=predictions_dict,
            iou_dict=IOU_DICT,  # Для не-SKIP классов
            launch_app_on_completion=LAUNCH_APP_INTERACTIVELY,
            port=port_for_this_dataset,
        )
        if dataset:
            loaded_datasets_names.append(dataset.name)
            if LAUNCH_APP_INTERACTIVELY:
                print(
                    f"Сессия для {dataset.name} запущена. Нажмите Ctrl+C в консоли, где запущен Python, чтобы остановить ЕЁ и перейти к следующему, ИЛИ закройте вкладку браузера."
                )
                # session.wait() # Это заблокирует выполнение до закрытия окна FiftyOne
                input(
                    "Нажмите Enter для обработки следующего файла (если сессия не блокирует)..."
                )  # Дает время посмотреть

    print("\n=== Обработка всех CSV завершена ===")
    if loaded_datasets_names:
        print("Следующие датасеты были созданы или обновлены в FiftyOne:")
        for name in loaded_datasets_names:
            print(f"- {name}")
        print(
            "\nВы можете запустить FiftyOne App вручную, чтобы их просмотреть, например:"
        )
        print("import fiftyone as fo")
        print(
            f"dataset = fo.load_dataset('{loaded_datasets_names[0]}') # Загрузить один из датасетов"
        )
        print("session = fo.launch_app(dataset)")
        print("session.wait()")
    else:
        print("Не было создано ни одного датасета.")


if __name__ == "__main__":
    main()
