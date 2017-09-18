# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy
from matplotlib import pyplot


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")  # Single input, single output
    print(sys.argv[
              0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, three outputs
    print(sys.argv[
              0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_helper(channel):

    maximum_pixel_val = 256
    channel_hist, channel_bins = numpy.histogram(channel.flatten(), maximum_pixel_val, [0, maximum_pixel_val])
    cdf = channel_hist.cumsum()

    cdf = cdf * channel_hist.max() / cdf.max() #normalize the Cummulative distribution function so that it fits in the plot with histogram

    # plot the histogram reference from http://opencvpython.blogspot.com/2013/03/histograms-2-histogram-equalization.html
    # pyplot.plot(cdf, color ='b')
    # pyplot.hist(channel.flatten(), maximum_pixel_val, [0, maximum_pixel_val], color = 'r')
    # pyplot.xlim([0, maximum_pixel_val])
    # pyplot.legend(('cdf', 'histogram'), loc = 'upper right')
    # pyplot.show()

    # Apply equalization on Histogram
    # Get minimum from the cdf (excluding 0)

    cdf_modified = numpy.ma.masked_equal(cdf, 0) # This will remove all 0 from CDF
    cdf_min = cdf_modified.min()
    cdf_max = cdf_modified.max()
    cdf_modified = (maximum_pixel_val-1) * (cdf_modified - cdf_min) / (cdf_max-cdf_min)

    #refill all the previously removed 0
    cdf = numpy.ma.filled(cdf_modified, 0).astype('uint8')

    # # plot the New histogram refered from http://opencvpython.blogspot.com/2013/03/histograms-2-histogram-equalization.html
    # pyplot.plot(cdf, color ='b')
    # pyplot.hist(channel.flatten(), maximum_pixel_val, [0, maximum_pixel_val], color = 'r')
    # pyplot.xlim([0, maximum_pixel_val])
    # pyplot.legend(('cdf', 'histogram'), loc = 'upper right')
    # pyplot.show()

    return cdf[channel]

def histogram_equalization(img_in):
    # Write histogram equalization here
    img_out = img_in  # Histogram equalization result

    # splitting images into BGR channels
    blue, green, red = cv2.split(img_in)
    equalized_channels = []
    equalized_channels_correct = []
    for channel in [blue, green, red]:
        equalized_channels.append(histogram_helper(channel))
        # equalized_channels_correct.append((cv2.equalizeHist(channel)))

    img_out = cv2.merge(equalized_channels)
    # img_actual = cv2.equalizeHist(img_in)

    # cv2.imshow('Original', img_in)
    # cv2.imshow('Hist Eq', img_out)
    #
    # cv2.imshow('Correct', cv2.merge(equalized_channels_correct))
    #
    # cv2.waitKey(0)

    return True, img_out


def Question1():
    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):

    # Write low pass filter here
    img_out = img_in  # Low pass filter result

    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # Take the fft of the image
    image_fourier_transform = numpy.fft.fft2(img_in)
    # shift the result by N/2 to bring zero frequency component to center
    image_fourier_transform_shifted = numpy.fft.fftshift(image_fourier_transform)

    # Create a mask to so that all the low freq pass but high freq are attenuated
    rows, cols= img_in.shape
    masklen = 20
    low_pass_mask = numpy.full_like(image_fourier_transform_shifted, 0)
    low_pass_mask[rows/2 - masklen: rows/2 + masklen, cols/2 - masklen : cols/2 + masklen ] = 1

    # Apply the low pass filter mask
    image_fourier_transform_shifted = image_fourier_transform_shifted * low_pass_mask

    # Time to get back the image!
    # Apply inverse shift
    inverse_image_fourier_transform = numpy.fft.ifftshift(image_fourier_transform_shifted)
    img_out = numpy.fft.ifft2(inverse_image_fourier_transform)
    img_out = numpy.abs(img_out)

    return True, img_out


def high_pass_filter(img_in):
    # Write high pass filter here
    img_out = img_in  # High pass filter result

    # convert to grayscale
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # Take FFT and shift the zero frequency component (DC component) to center
    image_fourier_transform = numpy.fft.fft2(img_in)
    image_fourier_transform_shifted = numpy.fft.fftshift(image_fourier_transform)

    # Create High pass mask which will attenuate lower freq
    rows, cols = img_in.shape
    masklen = 20
    high_pass_mask = numpy.full_like(image_fourier_transform, 1)
    high_pass_mask[rows/2 - masklen: rows/2 + masklen, cols/2 - masklen : cols/2 + masklen ] = 0

    # Apply the High pass filter
    image_fourier_transform_shifted = image_fourier_transform_shifted * high_pass_mask

    # TIme to get back the image

    inverse_shifted_fft_image = numpy.fft.ifftshift(image_fourier_transform_shifted)
    img_out= numpy.fft.ifft2(inverse_shifted_fft_image)
    img_out = numpy.abs(img_out)

    return True, img_out


def deconvolution(img_in):
    # Write deconvolution codes here
    img_out = img_in  # Deconvolution result

    # img_in = cv2.imread("blurred2.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # convert to grayscale
    # img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # Take FFT and shift the zero frequency component (DC component) to center
    image_fourier_transform = numpy.fft.fft2(img_in)
    image_fourier_transform_shifted = numpy.fft.fftshift(image_fourier_transform)

    # Applying a gaussian kernel for deconvolution
    rows, cols = img_in.shape
    gaussian_kernel = cv2.getGaussianKernel(21, 5)
    gaussian_kernel = gaussian_kernel*gaussian_kernel.T

    fourier_gaussian = numpy.fft.fft2(gaussian_kernel, (rows, cols))
    shifted_fourier_gaussian = numpy.fft.fftshift(fourier_gaussian)

    # Apply the gaussian kernel
    image_fourier_transform_shifted = image_fourier_transform_shifted / shifted_fourier_gaussian

    # TIme to get back the image
    inverse_shifted_fft_image = numpy.fft.ifftshift(image_fourier_transform_shifted)
    img_out= numpy.fft.ifft2(inverse_shifted_fft_image)
    img_out = numpy.abs(img_out)
    img_out *= 255

    if img_out.dtype != numpy.uint8:
        img_out.astype(numpy.uint8)

    return True, img_out

def Question2():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):
    # Write laplacian pyramid blending codes here
    img_out = img_in1  # Blending result

    return True, img_out


def Question3():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
