
import os
import shutil

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

import main

# Подключение файла для работы с моделью и предсказаний
from identification import *


# Глобальный статус
GLOBAL_STATE = 0

# 1 = Uploading | 2 = Processed
ACTIVE_LIST = 0

# Пути к каталогам с изображениями
uploading_images_path = "Work/UploadingImages"
processed_images_path = "Work/ProcessedImages"


class UIFunctions(main.MainApplication):

    def __init__(self):
        super().__init__()

    @property
    def status():
        return GLOBAL_STATE

    @status.setter
    def status(self, stat):
        global GLOBAL_STATE
        GLOBAL_STATE = stat

    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == 0:
            self.showMaximized()
            GLOBAL_STATE = 1
        else:
            GLOBAL_STATE = 0
            self.showNormal()
            self.resize(self.width(), self.height())

    def uiDefinitions(self):

        # Обработка нажатия на кнопку скрытия приложения
        self.button_hide.clicked.connect(lambda: self.showMinimized())

        # Обработка нажатия на кнопку максимизации окна приложения
        self.button_minimize.clicked.connect(
            lambda: UIFunctions.maximize_restore(self))

        # Обработка нажатия на кнопку закрытия приложения
        self.button_close.clicked.connect(lambda: self.close())

        # Обработка нажатий на кнопки очистки списков
        self.button_clear_processed.clicked.connect(
            lambda: clear_processed_list(self, self.list_processed_materials))
        self.button_clear_uploading.clicked.connect(
            lambda: clear_uploading_list(self, self.list_uploading_materials))

        # Обработка нажатий на кнопки удаления выделенных элементов
        self.button_delete_processed.clicked.connect(
            lambda: deleteItemsFromProcessedList(self, self.list_processed_materials))
        self.button_delete_uploading.clicked.connect(
            lambda: deleteItemsFromUploadingList(self, self.list_uploading_materials))

        # Обработка нажатия на кнопку добавления файлов для обработки
        self.button_add_uploading.clicked.connect(
            lambda: getFiles(self)
        )

        # Обработка нажатия на кнопку открытия папки с обработанными изображениями
        self.button_open_folder_process.clicked.connect(
            lambda: openFolder()
        )

        self.button_previous_image.clicked.connect(
            lambda: loadPreviusImage(self))
        self.button_next_image.clicked.connect(lambda: loadNextImage(self))

        self.list_processed_materials.itemClicked.connect(
            lambda: switchListFocus(self, 2))
        self.list_uploading_materials.itemClicked.connect(
            lambda: switchListFocus(self, 1))

        self.button_start_processing.clicked.connect(
            lambda: image_processing(uploading_images_path)
        )


def printImage(self):
    global ACTIVE_LIST
    if ACTIVE_LIST == 1:
        content = self.list_uploading_materials.currentItem()
        if content is None or content == "":
            return
        filename = content.text()
        self.label_photo_preview.setPixmap(
            QPixmap(f"{uploading_images_path}/{filename}"))

    elif ACTIVE_LIST == 2:
        content = self.list_processed_materials.currentItem()
        if content is None or content == "":
            return
        filename = content.text()
        self.label_photo_preview.setPixmap(
            QPixmap(f"{processed_images_path}/{filename}"))


def switchListFocus(self, list_number: int):
    printImage(self)
    global ACTIVE_LIST
    ACTIVE_LIST = list_number


# 1 = Uploading | 2 = Processed
def loadPreviusImage(self):
    global ACTIVE_LIST
    if ACTIVE_LIST == 1:
        if self.list_uploading_materials.currentRow() == 0 or self.list_uploading_materials.count() == 0:
            return
        self.list_uploading_materials.setCurrentRow(
            self.list_uploading_materials.currentRow() - 1)
        printImage(self)
    elif ACTIVE_LIST == 2:
        if self.list_processed_materials.currentRow() == 0 or self.list_processed_materials.count() == 0:
            return
        self.list_processed_materials.setCurrentRow(
            self.list_processed_materials.currentRow() - 1)
        printImage(self)


# 1 = Uploading | 2 = Processed
def loadNextImage(self):
    global ACTIVE_LIST
    if ACTIVE_LIST == 1:
        if self.list_uploading_materials.currentRow() == self.list_uploading_materials.count() - 1 or self.list_uploading_materials.count() == 0:
            return
        self.list_uploading_materials.setCurrentRow(
            self.list_uploading_materials.currentRow() + 1)
        printImage(self)
    elif ACTIVE_LIST == 2:
        if self.list_processed_materials.currentRow() == self.list_processed_materials.count() - 1 or self.list_processed_materials.count() == 0:
            return
        self.list_processed_materials.setCurrentRow(
            self.list_processed_materials.currentRow() + 1)
        printImage(self)


def clear_uploading_list(self, list_widget):
    list_widget.clear()
    self.label_photo_preview.clear()
    for file in os.listdir(uploading_images_path):
        if os.path.isfile(os.path.join(uploading_images_path, file)):
            os.remove(os.path.join(uploading_images_path, file))


def clear_processed_list(self, list_widget):
    list_widget.clear()
    self.label_photo_preview.clear()
    for file in os.listdir(processed_images_path):
        if os.path.isfile(os.path.join(processed_images_path, file)):
            os.remove(os.path.join(processed_images_path, file))


def updateUploadingList(list_widget):
    list_widget.clear()

    for file in os.listdir(uploading_images_path):
        list_widget.addItem(file)


def updateProcessingList(list_widget):
    list_widget.clear()

    for file in os.listdir(processed_images_path):
        list_widget.addItem(file)


def openFolder():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    os.startfile(f'{dirname}/{processed_images_path}/')


def getDirectory(self):
    return QFileDialog.getExistingDirectory(self, "Выбрать папку c изображениями", ".")


def getFiles(self):  # sourcery skip: use-named-expression
    directory_from = getDirectory(self)
    if directory_from == "":
        return
    directory_to = uploading_images_path
    for file in os.listdir(directory_from):
        file_path_from = f'{directory_from}/{file}'
        if os.path.isfile(file_path_from) and (file_path_from.endswith(('.jpeg', '.jpg', '.png'))):
            shutil.copy(file_path_from, directory_to)

    updateUploadingList(self.list_uploading_materials)


def deleteItemsFromUploadingList(self, list_widget):
    listItems = list_widget.selectedItems()
    for item in listItems:
        if os.path.isfile(os.path.join(uploading_images_path, item.text())):
            os.remove(os.path.join(uploading_images_path, item.text()))

    if not listItems:
        return
    for item in listItems:
        list_widget.takeItem(list_widget.row(item))

    self.label_photo_preview.clear()
    list_widget.clearSelection()


def deleteItemsFromProcessedList(self, list_widget):
    listItems = list_widget.selectedItems()
    for item in listItems:
        if os.path.isfile(os.path.join(processed_images_path, item.text())):
            os.remove(os.path.join(processed_images_path, item.text()))

    if not listItems:
        return
    for item in listItems:
        list_widget.takeItem(list_widget.row(item))

    self.label_photo_preview.clear()
    list_widget.clearSelection()
