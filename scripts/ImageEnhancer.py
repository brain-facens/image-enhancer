import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
import albumentations as A

class ImageEnhancer:
    def __init__(self, images_to_change_path: str, image_to_process_path: str, images_output_path: str) -> None:
        """
        This functions is responsible for 
        variables initialization
        
        ---
        
        Args:
            images_to_change_path (str): images path to change
            image_to_process_path (str): images path to process
            images_output_path (str): images path to save
        """
        self.images_to_change_path = images_to_change_path
        self.image_to_process_path = image_to_process_path
        self.images_output_path = images_output_path

        

    def rename_files(self) -> None:
        """ 
        This functions is responsible for 
        rename files and save it
        
        ---
        
        Args:
        
            
        Returns:
        
        """
        
        i = 0 
        for file in os.listdir(self.images_to_change_path):
            dest = f"{self.images_to_change_path}{i}.jpg" 
            source = f"{self.images_to_change_path}/{file}"
            os.rename(source, dest)
            i += 1
            
    def show_images(self, original: np.array, processed = False, isHist = False) -> None:
        """ 
        This functions is responsible for displaying 
        images on jupyter notebooks
        
        ---
        
        Args:
            original - original image
            processed - True () or False
            isHist - True or False
            
        Returns:
        
        """
        fig = plt.figure(figsize=(10, 9)) 
        fig.add_subplot(1, 2, 1) 
        
        plt.imshow(original, cmap='gray') 
        plt.title("Original") 
        plt.show()
        
        fig.add_subplot(1, 2, 2)
        if isHist:
            hist, bins = np.histogram(original.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()

            plt.plot(cdf_normalized, color = 'b')
            plt.hist(original.flatten(), 256, [0, 256], color = 'r')
            plt.xlim([0, 256])
            plt.legend(('cdf', 'histogram'), loc = 'upper left')
        if processed:
            plt.imshow(processed) 
            plt.title("Processed") 

        plt.show()
        
    def hist_equalization_cdf(self, image: np.array) -> np.array:
        """ 
        This functions is responsible for the 
        Histogram Equalization using cdf normalization
        
        ---
        
        Args:
            image - image reference to processing
            
        Returns:
            cdf[image] - image processed
        
        """
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        
        return cdf[image]

    def hist_equalization(self, image: np.array) -> np.array:
        """ 
        This functions is responsible for the 
        Histogram Equalization
        
        ---
        
        Args:
            image - image reference to processing
            
        Returns:
            dst - image processed
        
        """
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        dst = cv.equalizeHist(img)
        
        return dst

    def resize_img(self, image: np.array, w: int, h: int) -> np.array:
        """ 
        This functions is responsible for the 
        resize the image using openCv
        
        ---
        
        Args:
            image - image reference to processing
            w - image width
            h - image height     
            
        Returns:
            resized - image processed
        
        """
        dim = (w, h)
        resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
        
        return resized

    def binarize_image_CV(self, image: np.array) -> np.array:
        """ 
        This functions is responsible for the 
        binarize image using open cv image otsu threshold processing
        
        ---
        
        Args:
            image - image reference to processing
                    
        Returns:
            im_gray_th_otsu - image processed
        
        """
        im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        th, im_gray_th_otsu = cv.threshold(im_gray, 128, 192, cv.THRESH_OTSU)
        
        return im_gray_th_otsu

    def binarize_image_NP(self, image: np.array, thresh: float) -> np.array:
        """ 
        This functions is responsible for the 
        binarize image using numpy image processing
        
        ---
        
        Args:
            image - image reference to processing
            thresh - threshold value to apply
                    
        Returns:
            im_bin_keep - image processed
        
        """
        
        im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #im_gray = np.array(Image.open(f'{path}').convert('L'))
        im_bin_keep = (im_gray > thresh) * im_gray
        
        return im_bin_keep

    def apply_brightness_contrast(self, image: np.array, brightness = 0, contrast = 0) -> np.array:
        """ 
        This functions is responsible for the 
        apply brightness contrast image processing
        
        ---
        
        Args:
            image - image reference to processing
            brightness - brightness value to apply
            contrast - contrast value to apply
                    
        Returns:
            buf - image processed
        
        """
        
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv.addWeighted(image, alpha_b, image, 0, gamma_b)
        else:
            buf = image.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
        
    def dilatation(self, image: np.array, it: int, k: int) -> np.array:
        """ 
        This functions is responsible for the 
        Dilatation image processing
        
        ---
        
        Args:
            image - image reference to processing
            it - interations number
            k - value for kernel matrix
                    
        Returns:
            dilated_im - dilated image
        
        """
        kernel = np.ones((k, k), np.uint8)
        dilated_im = cv.dilate(image, kernel, anchor=(0,0), iterations = it)
        
        return dilated_im

    def clahe_cv(self, image: np.array) -> np.array:
        """ 
        This functions is responsible for the 
        Contrast Limited Adaptive Histogram Equalization
        
        ---
        
        Args:
            image - image reference to processing
                    
        Returns:
            gray_img1_clahe - image equalized
        
        """
        clahe = cv.createCLAHE(clipLimit = 20)
        gray_img1_clahe = clahe.apply(image)
        
        return gray_img1_clahe

            
    def augmentation(self, image: np.array, seed_val: int) -> np.array:
        """ 
        This functions is responsible for the 
        image augmentation generation
        
        ---
        
        Args:
            image - image reference to augmentation
            seed_val - value for variance
        
        Returns:
            transformed_image - transformed image 
        
        """
        
        transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.GaussNoise(),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
            ], p=0.3)
        ])
        
        random.seed(seed_val)
        transformed_image = transform(image=image)['image']
        
        return transformed_image


    def preprocessing_image(self, w: int, h: int) -> None:
        """ 
        This functions is responsible for the 
        image processing pipeline
        
        ---
        
        Args:
            w - image width
            h - image height
        
        Returns:
        
        """
        i = 0 
        seed = 0
        
        os.makedirs(self.images_output_path, exist_ok = True)

        for file in os.listdir(self.image_to_process_path):
            dest_img = f"{self.images_output_path}/{i}.jpg" 
            source = f"{self.image_to_process_path}/{file}"
            image = cv.imread(source)
            #image = self.resize_img(image, w, h)
            image = self.apply_brightness_contrast(image, 30, 100)
            #image = self.hist_equalization(image)
            #image = self.clahe_cv(image)
            #image = self.augmentation(image, seed)
            
            cv.imshow("Img", image)
            cv.imwrite(dest_img, image)
            
            if cv.waitKey(1) == ord('q'):
                break
            
            if seed == 100:
                seed = 0
            
            i += 1
            seed += 1
            
            
if __name__ == "__main__":
    images_to_change_path = "/home/nata-brain/Documents/proj/image-enhancer/datasets/cable_dataset_tester/images/dae_dataset/val/test/bad"
    image_to_process_path = "/home/nata-brain/Documents/proj/image-enhancer/datasets/cable_dataset_tester/images/dae_dataset/val/test/bad"
    images_output_path = "/home/nata-brain/Documents/proj/image-enhancer/datasets/cable_dataset_tester/images/dae_dataset/val/test/bad/pr" 
    
    processing = ImageEnhancer(images_to_change_path, image_to_process_path,images_output_path )
    processing.preprocessing_image(320, 320)
    