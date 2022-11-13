
import sys
import platform

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizeGrip, QStatusBar
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QEvent

import design  # Подключение файла дизайна
import Resources_rc  # Подключение файла ресурсов

# Подключение файла для работы с пользовательским интерфейсом
from ui_functions import *


class MainApplication(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()

        # Это нужно для инициализации нашего дизайна
        self.setupUi(self)
        # Убираю стандартную рамку операционной системы
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.print_status_bar(self.system_info())
        
        # Обновляю списки при запуске
        updateUploadingList(self.list_uploading_materials)
        updateProcessingList(self.list_processed_materials)
        
        # Очистка превью при загрузке
        self.label_photo_preview.clear()

        self.frame_spacer_custom_bar.mouseMoveEvent = self.moveWindow

        self.frame_spacer_custom_bar.mouseDoubleClickEvent = self.doubleClickMaximizeRestore

        # Загрузка функций для работы с интерфейсом
        UIFunctions.uiDefinitions(self)

        self.list_processed_materials.installEventFilter(self)
        self.list_uploading_materials.installEventFilter(self)

    def system_info(self):
        return f'System: {platform.system()} | ' \
              f'Architecture: {platform.architecture()[1]} | ' \
              f'Version: {platform.release()}'
              
    def print_status_bar(self, message):
        self.statusBar.showMessage(message)

    # Функция отслеживания изменений у объекта
    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            if obj == self.list_processed_materials:
                self.label_image_status.setText("Результат")
                self.list_uploading_materials.clearSelection()

            elif obj == self.list_uploading_materials:
                self.label_image_status.setText("Исходное изображение")
                self.list_processed_materials.clearSelection()

        return super(MainApplication, self).eventFilter(obj, event)

    # Функция, отвечающая за передвижение окна за кастомную рамку
    def moveWindow(self, event):
        # Если окно максимального размера, то переключить на нормальное
        if UIFunctions.status == 1:
            UIFunctions.maximize_restore(self)

        # Передвижение окна
        if event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.globalPos() - self.dragPos)
            self.dragPos = event.globalPos()
            event.accept()

    # Функция, отвечающая за отслеживание нажатий мышкой по окну приложения
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    # Функция, отвечающая за обработку двойных нажатий
    def doubleClickMaximizeRestore(self, event):
        # При двойном нажатии меняем статус на развёрнутое окно
        if event.type() == QtCore.QEvent.MouseButtonDblClick:
            UIFunctions.maximize_restore(self)


def main():
    # Новый экземпляр QApplication
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()  # Создаём объект класса MainApplication
    window.show()  # Показываем окно
    app.exec_() # Запускаем цикл обработки событий


if __name__ == '__main__':  #
    main()
