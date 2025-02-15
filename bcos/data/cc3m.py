import os
import clip
import torch 
import numpy as np
import matplotlib.pyplot as plt
import webdataset as wds

from bcos.settings import CC3M_PATH

class CustomDataCollatorImg:
    def __init__(self) -> None:
        pass
        
    def __call__(self, batch):      
                
        imgs = [i["image"] for i in batch]
        idxs  =[i['__key__'] for i in batch]
        imgs = torch.stack(imgs)
        
        return imgs, idxs 
    

class CustomDataCollatorText:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        
    def __call__(self, batch):  
        
        idxs  =[i['__key__'] for i in batch]
        texts = [i['text'] for i in batch]
        
        texts = self.tokenizer(texts, truncate=True)
                
        return texts, idxs 


class CC3MText:
    def __init__(self):
        pass
        
    def get_wds_dataset(self, input_shards, batch_size, collator=None):
        """
        return a dataset that returns an image, and text
        """

        pipeline = [
            wds.SimpleShardList(input_shards),
            # at this point we have an iterator over all the shards
        ]

        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),
                # wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb"),
                wds.rename(text="txt"),
                wds.map_dict(text=lambda text: text),
            ]
        )
        pipeline.extend([wds.batched(batch_size, partial=False, collation_fn=collator)])
        dataset = wds.DataPipeline(*pipeline)
        return dataset

    def get_dataloader(self, dataset, batch_size= None, shuffle= False, num_workers=1):
        loader =  torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle= shuffle, num_workers=num_workers)
        return loader


class CC3MImg:
    def __init__(self):
        pass
        
    def get_wds_dataset(self, input_shards, transform, batch_size, collator=None):
        """
        return a dataset that returns an image, and text
        """

        pipeline = [
            wds.SimpleShardList(input_shards),
            # at this point we have an iterator over all the shards
        ]

        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),
                # wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb"),
                wds.rename(image="jpg;png;jpeg"),
                wds.map_dict(image=transform),
            ]
        )

        # if val == False:
            # pipeline.extend([wds.shuffle(100 * batch_size)])

        pipeline.extend([wds.batched(batch_size, partial=False, collation_fn=collator)])

        dataset = wds.DataPipeline(*pipeline)
        return dataset

    def get_dataloader(self, dataset, batch_size= None, shuffle= False, num_workers=1):
        loader =  torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= shuffle, num_workers=num_workers)
        return loader


if __name__=='__main__':

    def plot_and_save_image_array(image_array, captions, save_directory):
        """
        Plot and save 'k' number of images from an image array using matplotlib.

        Parameters:
        - image_array (numpy.ndarray): Array of images where each row represents an image.
        - captions (list): List of captions corresponding to each image.
        - save_directory (str): Directory where the images will be saved. If it doesn't exist, it will be created.
        """
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Number of images to plot
        k = len(captions)

        for i in range(k):
            # Create subplots
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))  # Adjust figsize as needed

            # Plot the image
            img = image_array[i]
            img= np.swapaxes(img,0,1)
            img= np.swapaxes(img,1,2)
            axes.imshow(img)
            axes.axis('off')  # Hide axes
            axes.set_title(captions[i])

            # Save the plot with a different index in the filename
            save_path = os.path.join(save_directory, f"image_{i + 1}.png")
            plt.savefig(save_path, bbox_inches='tight')


    # Get dataloader  #RN50, 'ViT-B/16'
    clip_name = "RN50"; device='cuda'; batch_size= 4096; num_workers = 8
    data_root= CC3M_PATH
    train_path = "training"; val_path = 'validation'

    train_shard = os.path.join(data_root,train_path,'{00000..00331}.tar')
    val_shard = os.path.join(data_root,val_path, '{00000..00001}.tar')
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    model = clip_model.visual

    print('Model loaded')
    print('starting to load data')

    # For Image
    collator = CustomDataCollatorImg()
    cc3m_obj = CC3MImg()

    val_dtset  = cc3m_obj.get_wds_dataset(val_shard, clip_preprocess, batch_size, collator=collator)
    val_loader = cc3m_obj.get_dataloader(val_dtset, batch_size= None, shuffle= False)
    
    idxs_img= []
    imgs=[]
    for i, (img, idx) in enumerate(val_loader):
        idxs_img.extend(idx)
        tmp_img= [i.numpy() for i in img]
        imgs.extend(tmp_img)

    print('imgs stored')
    print('starting to load text')

    # For Text
    collator= CustomDataCollatorText(clip.tokenize)
    cc3m_obj = CC3MText()
    val_dtset = cc3m_obj.get_wds_dataset(train_shard, batch_size, collator=collator)
    val_loader = cc3m_obj.get_dataloader(val_dtset, batch_size= None, shuffle= False)
    
    idxs_text= []
    texts=[]
    for i, (text, idx) in enumerate(val_loader):
        idxs_text.extend(idx)
        texts.extend(text)
        
    print('text stored')    
    
    assert(idxs_text== idxs_img)
    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    idxs_text.sort()
    idxs_img.sort()
    assert(idxs_text== idxs_img)

    perm = np.random.permutation(np.arange(len(imgs)))
    texts= np.array(texts)
    idxs_text= np.array(idxs_text)
    imgs= np.array(imgs)

    # idxs_plot= idxs_text[perm]
    imgs_plot= imgs[perm][:20]
    texts_plot= texts[perm][:20]
   
    # Plot and see after extracting the image,caption pairs match (i.e., there is no shuffling)
    plot_and_save_image_array(imgs_plot, texts_plot, './vis')


    
