import time
import re
import logging
import sys
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Настройка логирования с правильной кодировкой
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("profi_bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Текст для отклика
RESPONSE_TEXT = """Здравствуйте! Позвольте представиться, меня зовут Мусин Даниил Денисович. Я выпускник физического факультета МГУ, сдал ЕГЭ по физике на 96 баллов. Работаю в лаборатории РМА и СИ Российской Академии Наук.\
\nЯ преподаю математику и физику и буду рад помочь вам в подготовке к экзаменам или олимпиадам. Мои ученики занимают призовые места и поступают в престижные лицеи. Также я готовлю к международным экзаменам на английском языке, подробная информация и отзывы в анкете.\
"""

# Нежелательные слова в пожеланиях
UNWANTED_WORDS = [
    "вакансия", 
    "нужен только преподаватель", 
    "хотим только к", 
    "онлайн-школа", 
    "в команду", 
    "работа", 
    "требуется", 
    "должность", 
    "требуется", 
    "сотрудник", 
    "репетитор по вакансии", 
    "должность", 
    "нужен для школы"
]

# Минимальная цена для отклика
MIN_PRICE = 3000

def extract_price(price_text):
    """Извлекает числовое значение цены из текста"""
    if not price_text:
        return 0
    
    # Извлекаем числа из строки
    numbers = re.findall(r'\d+', price_text)
    if not numbers:
        return 0
    
    # Берем первое число как минимальную цену
    return int(numbers[0])

def check_unwanted_words(text):
    """Проверяет наличие нежелательных слов в тексте"""
    if not text:
        return False
    
    text = text.lower()
    for word in UNWANTED_WORDS:
        if word.lower() in text:
            logging.info(f"Найдено нежелательное слово: {word}")
            return True
    
    # Проверка на цену до 3000
    price_patterns = [r'цена до 3000', r'до 3000 ₽', r'не более 3000']
    for pattern in price_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logging.info("Найдено ограничение по цене до 3000")
            return True
    
    return False

def check_exists_by_xpath(xpath, driver):
    """Проверяет существование элемента по xpath"""
    try:
        element = driver.find_element(By.XPATH, xpath)
        return True
    except:
        return False

def handle_tariff_selection(driver):
    """Обработка выбора тарифа - два варианта"""
    try:
        # Сохраняем скриншот для диагностики
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshots/tariff_page_{timestamp}.png"
        os.makedirs("screenshots", exist_ok=True)
        driver.save_screenshot(screenshot_path)
        logging.info(f"Скриншот страницы тарифа сохранен: {screenshot_path}")
        
        # Вариант 1: Сначала нажать на тариф "Комиссия", потом на "Продолжить"
        try:
            # Ищем тариф "Комиссия" по тексту
            commission_tariff_selectors = [
                '//div[contains(text(), "Комиссия")]',
                '//span[contains(text(), "Комиссия")]',
                '//p[contains(text(), "Комиссия")]',
                '//*[contains(text(), "Комиссия")]'
            ]
            
            tariff_selected = False
            for selector in commission_tariff_selectors:
                try:
                    tariff_element = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    tariff_element.click()
                    logging.info(f"Тариф 'Комиссия' выбран селектором: {selector}")
                    tariff_selected = True
                    time.sleep(1)
                    break
                except:
                    continue
            
            if not tariff_selected:
                # Пробуем выбрать по радиокнопке или чекбоксу
                radio_selectors = [
                    '//input[@type="radio"]',
                    '//div[contains(@class, "radio")]',
                    '//div[contains(@class, "checkbox")]'
                ]
                
                for selector in radio_selectors:
                    try:
                        radio_elements = driver.find_elements(By.XPATH, selector)
                        for radio in radio_elements:
                            if radio.is_displayed() and radio.is_enabled():
                                radio.click()
                                logging.info(f"Радиокнопка выбрана селектором: {selector}")
                                tariff_selected = True
                                time.sleep(1)
                                break
                        if tariff_selected:
                            break
                    except:
                        continue
            
            # Теперь нажимаем "Продолжить"
            continue_selectors = [
                '//button[contains(text(), "Продолжить")]',
                '//button[contains(@class, "Button") and contains(text(), "Продолжить")]',
                '//div[contains(text(), "Продолжить")]',
                '//span[contains(text(), "Продолжить")]',
                '//*[contains(text(), "Продолжить")]'
            ]
            
            continue_clicked = False
            for selector in continue_selectors:
                try:
                    continue_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    continue_button.click()
                    logging.info(f"Кнопка 'Продолжить' нажата селектором: {selector}")
                    continue_clicked = True
                    time.sleep(2)
                    break
                except:
                    continue
            
            if continue_clicked:
                logging.info("Вариант 1 успешно выполнен: тариф выбран и 'Продолжить' нажато")
                return True
            
        except Exception as e:
            logging.warning(f"Вариант 1 не сработал: {str(e)}")
        
        # Вариант 2: Сразу нажать на "Продолжить" (если тариф уже выбран)
        try:
            logging.info("Пробуем вариант 2: сразу нажать 'Продолжить'")
            
            continue_selectors_v2 = [
                '//button[contains(text(), "Продолжить")]',
                '//div[contains(@class, "ButtonStyles") and contains(text(), "Продолжить")]',
                '//button[contains(@class, "button") and contains(text(), "Продолжить")]',
                '//a[contains(text(), "Продолжить")]',
                '//*[@role="button" and contains(text(), "Продолжить")]'
            ]
            
            for selector in continue_selectors_v2:
                try:
                    continue_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    continue_button.click()
                    logging.info(f"Вариант 2: Кнопка 'Продолжить' нажата селектором: {selector}")
                    time.sleep(2)
                    return True
                except:
                    continue
                    
        except Exception as e:
            logging.warning(f"Вариант 2 не сработал: {str(e)}")
        
        # Если ничего не сработало, пробуем найти любую кнопку на странице
        try:
            logging.info("Пробуем найти любую активную кнопку на странице")
            buttons = driver.find_elements(By.TAG_NAME, 'button')
            for i, button in enumerate(buttons):
                try:
                    if button.is_displayed() and button.is_enabled():
                        button_text = button.text.strip()
                        logging.info(f"Найдена кнопка #{i+1}: '{button_text}'")
                        if button_text:  # Если у кнопки есть текст
                            button.click()
                            logging.info(f"Нажата кнопка: '{button_text}'")
                            time.sleep(2)
                            return True
                except:
                    continue
        except Exception as e:
            logging.error(f"Не удалось найти ни одной кнопки: {str(e)}")
        
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при обработке выбора тарифа: {str(e)}")
        return False

max_attempts = 5000
count = 0

while True:
    count += 1
    
    try:
        logging.info(f"Попытка {count} из {max_attempts}")
        
        # Открываем страницу
        options = webdriver.ChromeOptions()
        options.binary_location = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
        options.add_argument("user-data-dir=C:/Users/Daniil/AppData/Local/Google/Chrome/User Data/Default_Profi")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)
        url = "https://profi.ru/backoffice/n.php"
        driver.get(url)
        
        time.sleep(1.5)
        
        # Позиция первого заказа
        x = 650
        y = 300
        actions = ActionChains(driver)
        actions.move_by_offset(x, y)
        actions.click()
        actions.perform()
        logging.info("Клик по первому заказу выполнен")
        
        time.sleep(1)
        
        # Переключаемся на новую вкладку
        all_windows = driver.window_handles
        if len(all_windows) > 1:
            driver.switch_to.window(all_windows[1])
            logging.info("Переключились на новую вкладку")
        else:
            logging.warning("Не открылась новая вкладка")
            driver.quit()
            time.sleep(50)
            continue
        
        # Сохраняем скриншот новой вкладки
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshots/new_tab_{timestamp}.png"
        os.makedirs("screenshots", exist_ok=True)
        driver.save_screenshot(screenshot_path)
        logging.info(f"Скриншот сохранен: {screenshot_path}")
        
        # Проверяем цену заказа
        try:
            price_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "TagStyles__TagText-sc-1m6uju1-4"))
            )
            price_text = price_element.text
            price = extract_price(price_text)

            logging.info(f"Найденная цена: {price}")

            if price <= MIN_PRICE:
                logging.info(f"Цена {price} меньше или равна минимальной {MIN_PRICE}. Пропускаем заказ.")
                driver.quit()
                time.sleep(50)
                continue
        except NoSuchElementException as e:
            logging.warning(f"Не удалось найти элемент с ценой на странице: {str(e)}")
            driver.quit()
            time.sleep(50)
            continue
        except Exception as e:
            logging.error(f"Ошибка при определении цены: {str(e)}")
            driver.quit()
            time.sleep(50)
            continue
        
        # Проверяем текст заказа на нежелательные слова
        try:
            wishes_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "bo_text"))
            )
            wishes_text = wishes_element.text
            logging.info(f"Текст пожеланий: {wishes_text}")

            if check_unwanted_words(wishes_text):
                logging.info("В пожеланиях найдены нежелательные слова. Пропускаем заказ.")
                driver.quit()
                time.sleep(50)
                continue
        except Exception as e:
            logging.error(f"Ошибка при проверке текста пожеланий: {str(e)}")
            driver.quit()
            time.sleep(50)
            continue
        
        # НОВЫЙ БЛОК: Обработка выбора тарифа
        if not handle_tariff_selection(driver):
            logging.error("Не удалось выбрать тариф или нажать 'Продолжить'")
            # Сохраняем скриншот ошибки
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/tariff_error_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            logging.info(f"Скриншот ошибки тарифа сохранен: {screenshot_path}")
            driver.quit()
            time.sleep(50)
            continue
        
        # Ждем загрузки формы отклика после выбора тарифа
        time.sleep(3)
        
        # Сохраняем скриншот после выбора тарифа
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshots/after_tariff_{timestamp}.png"
        driver.save_screenshot(screenshot_path)
        logging.info(f"Скриншот после выбора тарифа сохранен: {screenshot_path}")
        
        # Находим текстовое поле для отклика
        try:
            xpath_to_check = '//textarea[@placeholder="Уточните детали задачи или предложите свои условия"]'
            if check_exists_by_xpath(xpath_to_check, driver):
                textarea_element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, xpath_to_check))
                )
                textarea_element.click()
                time.sleep(0.5)
                textarea_element.clear()
                time.sleep(0.5)
                textarea_element.send_keys(RESPONSE_TEXT)
                logging.info("Текст отклика введен")
                time.sleep(1)
            else:
                logging.warning("Текстовое поле для отклика не найдено")
                # Сохраняем скриншот
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshots/no_textarea_{timestamp}.png"
                driver.save_screenshot(screenshot_path)
                logging.info(f"Скриншот сохранен: {screenshot_path}")
                driver.quit()
                time.sleep(50)
                continue
        except Exception as e:
            logging.error(f"Ошибка при вводе текста отклика: {str(e)}")
            driver.quit()
            time.sleep(50)
            continue
        
        # Попытка отправки сообщения - используем несколько методов
        send_button_found = False
        try:
            # Метод 1: По CSS селектору (оригинальный метод)
            send_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'p.ButtonStyles__Label-sc-1phciut-4.inzmmG'))
            )
            send_button.click()
            logging.info("Кнопка отправки нажата (метод 1: CSS селектор)")
            send_button_found = True
        except Exception as e1:
            logging.warning(f"Не удалось найти кнопку отправки методом 1: {str(e1)}")
            try:
                # Метод 2: По тексту кнопки
                send_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Отправить")]'))
                )
                send_button.click()
                logging.info("Кнопка отправки нажата (метод 2: по тексту)")
                send_button_found = True
            except Exception as e2:
                logging.warning(f"Не удалось найти кнопку отправки методом 2: {str(e2)}")
                try:
                    # Метод 3: По типу кнопки
                    send_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
                    )
                    send_button.click()
                    logging.info("Кнопка отправки нажата (метод 3: по типу)")
                    send_button_found = True
                except Exception as e3:
                    logging.warning(f"Не удалось найти кнопку отправки методом 3: {str(e3)}")
                    try:
                        # Метод 4: Перебор всех кнопок
                        buttons = driver.find_elements(By.TAG_NAME, 'button')
                        for i, button in enumerate(buttons):
                            try:
                                if button.is_displayed() and button.is_enabled():
                                    logging.info(f"Найдена видимая кнопка #{i+1}: {button.text}")
                                    button.click()
                                    logging.info(f"Нажата кнопка #{i+1}")
                                    send_button_found = True
                                    break
                            except:
                                continue
                    except Exception as e4:
                        logging.error(f"Не удалось найти ни одной кнопки для отправки: {str(e4)}")
        
        if not send_button_found:
            logging.error("Не удалось найти и нажать кнопку отправки сообщения")
            driver.quit()
            time.sleep(50)
            continue
        
        # Короткое ожидание после нажатия на кнопку
        time.sleep(5)
        
        # Завершаем цикл итерации
        time.sleep(5)
        driver.quit()
        logging.info(f"Итерация {count} завершена. Ожидание 50 секунд перед следующей попыткой.")
        time.sleep(50)

    except TimeoutException as e:
        logging.error(f"Ошибка тайм-аута загрузки страницы: {str(e)}")
        try:
            driver.quit()
        except:
            pass
        time.sleep(50)
        continue
    
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {str(e)}")
        try:
            driver.quit()
        except:
            pass
        time.sleep(50)
        continue
