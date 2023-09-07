import cv2
import imutils
import pytesseract
import re
from model import *
from datetime import datetime
from database import Database
import numpy as np

database = Database()

def processImage(originalImage):
    try:
        receipt = getReceipt(originalImage.copy())
        gray, thresholded = getThresholdedImage(receipt.copy())
        lower = [22, 30, 30]
        upper = [111, 255, 255]
        masked, hsv = segmentImage(receipt.copy(), lower, upper)
        denoised = removeNoise(masked.copy())
        # Apply the mask to the thresholded original image
        masked_image = cv2.bitwise_and(thresholded.copy(), thresholded.copy(), mask = denoised)
        contouredImage, boundingBox, imageCounters = drawBoundary(receipt.copy(), masked.copy(), 100)
        if not imageCounters:
            regions = [receipt.copy()]
        else:
            regions = getRegion(masked_image.copy(), imageCounters)
        data = []
        res = None
        for region in regions:
            products, tax, total = processRegion(region)
            if products:
                data.extend(products)
        if data:
            model = Expense(category = 'Groceries', date = datetime.now(), tax = tax, total = total, items = data)
            database.save(model)   
            res = model             
    except:
        raise Exception()
    return res
    

def getReceipt(image):
    height, width, _ = image.shape
    ratio = 500.0 / width 
    dim = (500, int (height * ratio))
    image = cv2.resize(image, dim)

    # grayscale the image and look for receipt edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11,), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, kernel)
    _, thresholded = cv2.threshold(dilated, 160, 255, cv2.THRESH_BINARY)

    # detect contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    edge = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.032 * peri, True)
        if len(approx) == 4:
            edge = approx
            break
    
    if edge is None:
        raise Exception()
    
    edge = np.array(sorted(edge, key=lambda point: (point[0][0], point[0][1])))
    source_points = np.float32(edge)
    result = four_point_transform(image, source_points)
    return result

def four_point_transform(image, rect):
    rect = rect.reshape(4, 2)
    (tl, bl, tr, br) = rect

    # calculate the width and height of the new image.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # set the destination points to get a top-down view of the image.
    # Order - top-left, top-right, botton-right, bottom-left
    dst = np.array([
		[0, 0],
        [0, maxHeight - 1],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1]], dtype = "float32")
    
	# compute the perspective transform matrix, apply it
    transformationMatrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transformationMatrix, (maxWidth, maxHeight))
	# return the warped image
    return warped


def getThresholdedImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 6)
    return gray, thresholded

def segmentImage(image, lowerThreshold, upperThreshold):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lowerThreshold, np.uint8) 
    upper = np.array(upperThreshold, np.uint8) 

    # Color segmentation with the given threshold ranges
    masked = cv2.inRange(hsv, lower, upper)

    return masked, hsv

def removeNoise(image):
    # Use morphological transformation to remove noise from the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = 4)
    return denoised

def drawBoundary(image, mask, thresholdArea):
    # Detect contours in the masked image
    contours, hierarchy, = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contouredImage = image.copy()
    boudingBox = image.copy()
    imageContours = []
    # Sort image countours
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for index, contour in enumerate(contours):
        # Skip small contours because its probably noise
        if  cv2.contourArea(contour) < thresholdArea:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.002 * peri, True)
        imageContours.append([len(approx), cv2.contourArea(contour), approx, contour])
        cv2.drawContours(contouredImage, contours, index, (0, 0, 255), 2, cv2.LINE_4, hierarchy)

        # Get bounding box position
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box in blue
        cv2.rectangle(boudingBox, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA, 0)
    return contouredImage, boudingBox, imageContours

def getRegion(image, contours):
    regionList = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c[3])
        regionList.append(image[y: y + h, x: x + w])
    return regionList

def processRegion(image):
    options = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(image.copy(), lang = 'eng', config = options)
    print(text)
    pricePattern = r'([0-9]+\.[0-9]+)'
    res = []
    for row in text.split("\n"):
	    if re.search(pricePattern, row) is not None:
                 res.append(row)

    product_pattern = re.compile(r'([0-9A-Z\s]+) ([\d.-]+)', re.IGNORECASE)
    prices_pattern = re.compile(r'\b\d+\.\d+|\b\d+\b')
    total_pattern = re.compile(r'TOTAL \$?([\d.]+)', re.IGNORECASE)
    tax_pattern = re.compile(r'TAX \$?([\d.]+)', re.IGNORECASE)
    products = []
    total = None
    tax = None
    for row in res:
        total_match = total_pattern.search(row)
        if total_match:
              total = float(total_match.group(1))
              continue
        tax_match = tax_pattern.search(row)
        if tax_match:
              tax = float(total_match.group(1))
              continue
        product_matches = product_pattern.findall(row)
        if product_matches:
             prices_match = prices_pattern.findall(row)
             products.append(([match[0] for match in product_matches], prices_match[-1]))
    return products, tax, total