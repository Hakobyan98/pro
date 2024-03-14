from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.options import Options
import cv2
from collections import Counter 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 


def initialize_driver(url):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver

def capture_screenshot(driver, filename):
    driver.get_screenshot_as_file(filename)

def process_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (900, 600), interpolation=cv2.INTER_AREA)
    image = image.reshape(image.shape[0] * image.shape[1], 3)
    return image

def find_top_colors(image):
    classes = KMeans(n_clusters=10, n_init=10)
    color_labels = classes.fit_predict(image)
    centroids = classes.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [centroids[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    return hex_colors, counts.values()

def save_color_palette(hex_colors, counts, filename):
    plt.figure(figsize=(12, 8))
    plt.pie(counts, colors=hex_colors)
    plt.savefig(filename)

def rgb_to_hex(rgb):
    hex_code = "#"
    for i in rgb:
        hex_code += ("{:02x}".format(int(i)))
    return hex_code

def print_colors(hex_colors):
    print("Found the following colors:\n")
    for color in hex_colors:
        print(color)


# The task is to write a script that extracts the color palette from the screenshot of the given url

# Main code
url = 'https://www.zoosp.ai'
screenshot_filename = "screenshot.png"
palette_filename = "screen_palette.png"

driver = initialize_driver(url)
sleep(1)

capture_screenshot(driver, screenshot_filename)
driver.quit()

image = process_image(screenshot_filename)

hex_colors, counts = find_top_colors(image)

save_color_palette(hex_colors, counts, palette_filename)
print_colors(hex_colors)