# Introduction to Computer Vision
# Author: Shreyas Bhatia (shrbhatia@cs.stonybrook.edu)
# Python Version: 2.7
# File Name hw0.py


# HW0: Hello Vision World
# Write an OpenCV program to do the following things:
# Read an image from a file and display it to the screen
# Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
# Resize the image uniformly by half

import cv2
import numpy

# Part1 Reading and displaying an image

imagepath = 'D:\IMG_0403.JPG'
image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
if image is None:
    print 'Image cannot be read. Please check the path'
    exit(0)
cv2.imshow('Introduction to Computer Vision', image)
cv2.waitKey(0)

# Part2: Add, Subtract, Multiply and Divide each pixel with scalar

# constants to be used
additionScalar = 50
subtractionScalar = 80
multiplicationScalar = 2
divisionScalar = 3

# Create dummy arrays for arithmetic operations
additionArray = numpy.full_like(image, additionScalar)
subtractionArray = numpy.full_like(image, subtractionScalar)
multiplicationArray = numpy.full_like(image, multiplicationScalar)
divisionArray = numpy.full_like(image, divisionScalar)

# Using openCV arithmetic functions to perform operations to make sure of saturated cast
resultAddition = cv2.add(image, additionArray)
resultSubtraction = cv2.subtract(image, subtractionArray)
resultMultiplication = cv2.multiply(image, multiplicationArray)
resultDivision = cv2.divide(image, divisionArray)

cv2.imshow('Addition', resultAddition)
cv2.waitKey(0)
cv2.destroyWindow('Addition')
cv2.imshow('Subtraction', resultSubtraction)
cv2.waitKey(0)
cv2.destroyWindow('Subtraction')
cv2.imshow('Multiplication', resultMultiplication)
cv2.waitKey(0)
cv2.destroyWindow('Multiplication')
cv2.imshow('Division', resultDivision)
cv2.waitKey(0)
cv2.destroyWindow('Division')

# Part 3: Resize the image uniformly by half
scaledImage = cv2.resize(image, None, 0, 0.5, 0.5)
cv2.imshow('Half', scaledImage)
cv2.waitKey(0)
cv2.destroyWindow('Half')

# Destroy the original window
cv2.destroyWindow('Introduction to Computer Vision')
