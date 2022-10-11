import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pandas as pd


def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """
    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)
    flag = True
    if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):

        if destination_path == None:
            destination_path = dataset_path
         
        
        def createList(Dir):
            #Traversing and storing directorys in list
            directory = []
            roots = []
            main_file = []
            for root,dir,file in os.walk(Dir):
                directory.append(dir)
                roots.append(root)
            
            #Ignoring the current directory
            roots = roots[1:]
            directory = directory[0]
            
            #making a full path by appending root to directory
            for i in range(len(directory)):
                label = ","+directory[i]
                for root1 , dir1, file1 in os.walk(roots[i]):
                    for names in file1:
                        data = roots[i]+"/"+names+label
                        main_file.append(data)

            return main_file

        dat = createList(DATASET_PATH)
       
        #Creating columns for pandas dataframe
        try:
            with open(destination_path+"dataset_attr.csv","w") as out_file:
                    out_file.write('path')
                    out_file.write(',')
                    out_file.write('label')                
                    out_file.write("\n")
        except IOError:
            print("Error writing file for dataframe columns ")
            flag = False
        
        #Writting file that will contain image paths
        try:
            with open(destination_path+"dataset_attr.csv","a") as out_file:
                for i in range(len(dat)):
                    out_s = str(dat[i])
                    out_file.write(out_s)
                    out_file.write("\n")
        except IOError:
                print("Error writing file for image paths ")
                flag = False
        
        # write out as dataset_attr.csv in destination_path directory
        # if no error
        return flag

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    
    if create_meta_csv(dataset_path, destination_path=destination_path):
        #reading main dataframe
        dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))
    
    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        #Resetting index as we have to split on index
        dframe = dframe.sample(frac=1).reset_index(drop=True)
        pass

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    
    s1 = np.floor(split_ratio * len(dframe)) #Conversion as index won't accept a float
    
    train_data = dframe.iloc[0:int(s1)]
    test_data = dframe.iloc[int(s1):]
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    # divide into train and test dataframes
    return train_data, test_data

class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.classes =  np.unique(data['label'])  # get unique classes from data dataframe

    def __len__(self):
        return len(self.data)
    
    
    def get_pilimage(self,path):
        """This funtion will retrive accept a image path, extract image and return pixel value for that image
            Args:
                path: Path to target image
            Return:
                1D vector of converted image pixels
        """
        
        img_file = Image.open(path)
        width, height = img_file.size
        img_color = img_file.convert('L')
        self.img_h = img_color.size[1]
        self.img_b = img_color.size[0]
        #Getting pixel values from image 
        value = np.asarray(img_color.getdata(), dtype=np.int).reshape((1,img_color.size[1], img_color.size[0]))
        #Conversion to 1D vector
        value = value.flatten()
    
        return value
    
    def get_label(self,idx,uni_list,data):
        """It will get the labels of image at particular index
           Args:
               idx:index
               uni_list: list containing unique labels of data
               data: image
           Return :
               class of the image
        """
        for i in range(len(uni_list)):
            if(data['label'][idx] == uni_list[i]):
                return i
            
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        image = self.get_pilimage(img_path) # load PIL image
        label = self.get_label(idx,self.classes,self.data) # get label (derived from self.classes; type: int/long) of image
        
        if self.transform:
            image = torch.FloatTensor(image.reshape(1,self.img_h,self.img_b))
            image = self.transform(image)
            
            
        return image, label




if __name__ == "__main__":
    # test config
    dataset_path = '../Data/fruits/'
    dest = './'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())
