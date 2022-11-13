
from keras import backend as K
import os
import cv2
import csv
import numpy as np
import pandas as pd
import collections
from tensorflow.keras.models import load_model

size = 2
image_shape_1 = 90
image_shape_2 = 300

MODEL_CATALOG = "Model"
MODEL_NAME = "model-0_88.h5"

# Пути к каталогам с изображениями
uploading_img_path = "Work/UploadingImages"
processed_img_path = "Work/ProcessedImages"


def prepareImg(imgName, height=90, width=300):
    """Возвращает изображение в формате, подходящем для дальнейшей работы в сети.

    Args:
        img (string): адрес изображения.
        height (int): итоговая высота изображения.
        width (int): итоговая ширина изображения
    """
    dim = (width, height)
    img = cv2.imread(imgName)
    resImg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    grayResImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
    fImg = np.frombuffer(grayResImg, dtype="u1", count=image_shape_1 *
                         image_shape_2).reshape(image_shape_1, image_shape_2)
    fImg = fImg[::size, ::size]

    # return grayResImg
    return fImg


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def this_is_it(img, indNum):
    """Буквально "Это оно?" Проверяет, принадлежит ли изображение особи, указанной вторым аргументом.

    Args:
        img (string): путь к изображению с его полным названием.
        indNum (int): номер особи для сравнения.

    Returns:
        float: возвращает "непохожесть".
    """

    disSum = 0

    # Указываем местоположение модели
    model = load_model(f"{MODEL_CATALOG}/{MODEL_NAME}",
                       custom_objects={'contrastive_loss': contrastive_loss})

    # У каждой особи 5 контрольных фотографий. Сравниваем с каждой и находим среднюю "непохожесть."
    for i in range(5):
        img1 = prepareImg("SN/1/1_0.jpg", 90, 300)
        img2 = prepareImg(f"individuals/{str(indNum)}/{str(i)}.jpg", 90, 300)

        # Что такое pair - не имею понятия, но автор использовал именно это для проверки в модели.
        pair = np.zeros([1, 2, 1, img1.shape[0], img1.shape[1]])
        pair[0, 0, 0, :, :] = img1
        pair[0, 1, 0, :, :] = img2
        pred = model.predict([pair[:, 0]/255, pair[:, 1]/255])

        disSum += pred[0][0]
    return disSum / 5


def affiliation(img):
    """Вычисляем весь список вероятностей.

    Args:
        img (string): путь к картинке.

    Returns:
        dictionary: словарь с вероятностями.
    """

    individs = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0, "20": 0,
                "21": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0, "28": 0, "29": 0, "30": 0,
                "31": 0, "32": 0, "33": 0, "34": 0, "35": 0, "36": 0, "37": 0, "38": 0, "39": 0, "40": 0,
                "41": 0, "42": 0, "43": 0, "44": 0, "45": 0, "46": 0, "47": 0, "48": 0, "49": 0, "50": 0,
                "51": 0, "52": 0, "53": 0, "54": 0, "55": 0, "56": 0, "57": 0, "58": 0, "59": 0, "60": 0,
                "61": 0, "62": 0, "63": 0, "64": 0, "65": 0, "66": 0, "67": 0, "68": 0, "69": 0, "70": 0,
                "71": 0, "72": 0, "73": 0, "74": 0, "75": 0, "76": 0, "77": 0, "78": 0, "79": 0, "80": 0,
                "81": 0, "82": 0, "83": 0, "84": 0, "85": 0, "86": 0, "87": 0, "88": 0, "89": 0, "90": 0,
                "91": 0, "92": 0, "93": 0, "94": 0, "95": 0, "96": 0, "97": 0, "98": 0, "99": 0, "100": 0,
                "101": 0, "102": 0}

    for i in range(1, 103):
        # Проходим по всем особям и сравниваем с введённой картинкой.
        dis = this_is_it(img, i)
        # Отнимаем непохожесть от 1го, чтобы получить вероятность, вместо непохожести.
        prob = 1 - dis
        # Устанавливаем этой особи установленную вероятность
        individs[str(i)] = prob

    return individs


def image_processing(path):

    sum_dictionary = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0,
                      "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0, "20": 0,
                      "21": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0, "28": 0, "29": 0, "30": 0,
                      "31": 0, "32": 0, "33": 0, "34": 0, "35": 0, "36": 0, "37": 0, "38": 0, "39": 0, "40": 0,
                      "41": 0, "42": 0, "43": 0, "44": 0, "45": 0, "46": 0, "47": 0, "48": 0, "49": 0, "50": 0,
                      "51": 0, "52": 0, "53": 0, "54": 0, "55": 0, "56": 0, "57": 0, "58": 0, "59": 0, "60": 0,
                      "61": 0, "62": 0, "63": 0, "64": 0, "65": 0, "66": 0, "67": 0, "68": 0, "69": 0, "70": 0,
                      "71": 0, "72": 0, "73": 0, "74": 0, "75": 0, "76": 0, "77": 0, "78": 0, "79": 0, "80": 0,
                      "81": 0, "82": 0, "83": 0, "84": 0, "85": 0, "86": 0, "87": 0, "88": 0, "89": 0, "90": 0,
                      "91": 0, "92": 0, "93": 0, "94": 0, "95": 0, "96": 0, "97": 0, "98": 0, "99": 0, "100": 0,
                      "101": 0, "102": 0}

    num_files = 0

    # Подготовка всех изображений в директории к обработке
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if os.path.isfile(file) and (file.endswith(('.jpeg', '.jpg', '.png'))):
            num_files += 1
            temp_image = prepareImg(file)

            # В переменную записываю результирующий словарь со всеми вероятностями
            res_dictionary = affiliation(temp_image)

            # Суммирую сум. словарь с словарём для текущего изображения
            for k, v in res_dictionary.items():
                sum_dictionary[k] += v

            # Другой метод суммирования (вроде более быстрый)
            # l = [sum_dictionary, res_dictionary]
            # counter = collections.Counter()
            # for d in l:
            #     counter.update(d)

            # Обновляю сум. словарь новой суммой (для другого метода)
            # sum_dictionary = dict(counter)

    # Предотвращаю возможное исключение деления на ноль
    if num_files == 0:
        return

    # Стандартизирую сум. словарь - делю каждую сумму на количество изображений, которые мы прошли

    for k, v in sum_dictionary.items():
        sum_dictionary[k] = v / num_files

    # Сортирую словарь по убыванию значения
    sort_dictionary = sorted(
        sum_dictionary.items(), key=lambda kv: kv[1], reverse=True)

    print(sort_dictionary)

    counter = 0
    probabilities = []
    for _, v in sort_dictionary.items():
        if counter > 4:
            break
        probabilities.append(v)

    cn = ["name", "top1", "top2", "top3", "top4", "top5"]
    df = pd.DataFrame(columns=["name", "top1", "top2", "top3", "top4", "top5"])


def affiliation2(img):
    """Вычисляем весь список вероятностей.

    Args:
        img (string): путь к картинке.

    Returns:
        dictionary: лист с вероятностями.
    """

    individs = [0 for _ in range(102)]

    for i in range(102):
        # Проходим по всем особям и сравниваем с введённой картинкой.
        dis = this_is_it(img, i)
        # Отнимаем непохожесть от 1го, чтобы получить вероятность, вместо непохожести.
        prob = 1 - dis
        # Устанавливаем этой особи установленную вероятность
        individs[i] = prob

    return individs


def csv_writer(inputDir, outputDir):
    """
    inputDir - Где лежат картинки для обработки?
    outputDir - Куда выложить результат обработки?

    whalesList - массив китов(массив массивов).
    whaleList - один кит в виде массива: имя(файла) и пять похожих на него в архиве.
    """

    whalesList = []

    for whale in os.listdir(inputDir):

        imgOfWhale = 0

        sum_probs_of_whale = [0 for _ in range(102)]

        for file in os.listdir(inputDir+"\\"+whale):
            for img in file:
                imgOfWhale += 1
                probs = affiliation2(inputDir+"\\"+whale+"\\"+img)

                for i in range(102):
                    sum_probs_of_whale[i] += probs[i]

        for i in range(102):
            sum_probs_of_whale[i] = sum_probs_of_whale[i] / imgOfWhale

        whaleList = [whale]

        for _ in range(5):
            topNow = probs.index(max(probs))+1
            probs[probs.index(max(probs))] = 0
            whaleList.append(topNow)

        whalesList.append(whaleList)

    csvFile = outputDir + "\\" + 'result.csv'

    with open(csvFile, 'w', newline='') as csvfile:
        fieldnames = ['name', 'top1', 'top2', 'top3', 'top4', 'top5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for whale in whalesList:
            writer.writerow({'name': str(whale[0]), 'top1': str(whale[1]), 'top2': str(
                whale[2]), 'top3': str(whale[3]), 'top4': str(whale[4]), 'top5': str(whale[5])})
