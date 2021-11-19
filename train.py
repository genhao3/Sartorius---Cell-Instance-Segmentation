import torch, torchvision,os,cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定第一块gpu

class DataGenerator(data.Dataset):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='/data_local/LJJ_Data/sartorius_dataset/train/',
                 dim=(256, 256), n_channels=3,
                 n_classes=3, random_state=2019, shuffle=True):
        self.dim = dim
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.on_epoch_end()

    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        ID = self.list_IDs[index]
        im_name = self.df['id'].iloc[ID]
        img_path = os.path.join(self.base_path,im_name+".png")

        img = self.__load_grayscale(img_path)

        mask = self.__load_y(im_name)

        return img.transpose(2, 0, 1).astype('float32'),mask.transpose(2, 0, 1).astype('float32')

        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    
    def __load_y(self, im_name):
        image_df = self.target_df[self.target_df['id'] == im_name]
        rles = image_df['annotation'].values
        masks = build_masks(rles,(520,704), colors=False)
        masks = cv2.resize(masks, (256, 256))
        #masks=masks.transpose(1,0)
        masks=np.expand_dims(masks, axis=-1)

        return masks
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # resize image
        dsize = (256, 256)
        img = cv2.resize(img, dsize)
        
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img



def train(train_df):
    train_idx, val_idx = train_test_split(train_df.index, random_state=2021, test_size=0.2)
    train_dataset = DataGenerator(
        train_idx, 
        df=train_df,
        target_df=train_df,
        n_classes=3
    )
    train_dataloaders = DataLoader(train_dataset,
                                    batch_size=48,
                                    shuffle=True,
                                    num_workers=6,
                                    pin_memory=True)
    val_dataset = DataGenerator(
        val_idx, 
        df=train_df,
        target_df=train_df,
        n_classes=3
    )
    val_dataloaders = DataLoader(val_dataset,
                                    batch_size=48,
                                    shuffle=True,
                                    num_workers=6,
                                    pin_memory=True)
    
    model = smp.Unet('efficientnet-b0', in_channels=1,classes=3, activation='sigmoid',encoder_weights='imagenet')
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.55),
        smp.utils.metrics.IoU(threshold=0.6),
        smp.utils.metrics.IoU(threshold=0.65),
        smp.utils.metrics.IoU(threshold=0.7),
        smp.utils.metrics.IoU(threshold=0.75),
        smp.utils.metrics.IoU(threshold=0.8),
        smp.utils.metrics.IoU(threshold=0.85),
        smp.utils.metrics.IoU(threshold=0.9),
        smp.utils.metrics.IoU(threshold=0.95),
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=1e-4),
    ])

    # 创建一个简单的循环，用于迭代数据样本
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device='cuda',
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device='cuda',
        verbose=True,
    )

    # 进行40轮次迭代的模型训练
    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloaders)
        valid_logs = valid_epoch.run(val_dataloaders)
		
		# 每次迭代保存下训练最好的模型
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            print('max_score',max_score)
            torch.save(model, './best_model.pth')
            print('Model saved!')

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(labels,input_shape, colors=True):
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask

def rle2maskResize(rle):
    # CONVERT RLE TO MASK 
    if (len(rle)==0): 
        return np.zeros((256,256) ,dtype=np.uint8)
    
    height= 520
    width = 704
    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    
    return mask.reshape( (height,width), order='F' )[::2,::2]


def show_png():
    train_df = pd.read_csv('/data_local/LJJ_Data/sartorius_dataset/train.csv')
    print(train_df.shape)
    print(train_df.head())

    sample_filename = '0030fd0e6378'
    sample_image_df = train_df[train_df['id'] == sample_filename]
    sample_path = "/data_local/LJJ_Data/sartorius_dataset/train/"+sample_image_df['id'].iloc[0]+".png"
    sample_img = cv2.imread(sample_path)
    sample_rles = sample_image_df['annotation'].values

    sample_masks1=build_masks(sample_rles,input_shape=(520, 704), colors=False)
    sample_masks2=build_masks(sample_rles,input_shape=(520, 704), colors=True)
    plt.imshow(sample_img)
    plt.savefig(r'sample_img.png')
    plt.imshow(sample_masks1[:,:,0])
    plt.savefig(r'sample_masks1.png')
    plt.imshow(sample_masks2)
    plt.savefig(r'sample_masks2.png')
if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())
    train_df = pd.read_csv('/data_local/LJJ_Data/sartorius_dataset/train.csv')
    train(train_df)