import tensorflow as tf
import os
from scipy.spatial.transform import Rotation
import numpy as np
import random
import matplotlib.image as mp

'''Base DatasetsLoader Class'''
class BaseDatasetsLoader:
    def __init__(self,name='BaseDatasetsLoader'):
        self.name = name
    def MakeTrainDatasets(self):
        print("make traindatasets")
    def MakeTestDatasets(self):
        print("make testdatasets")
    def GetTrainDatasets(self):
        print("get traindatasets")
    def GetTestDatasets(self):
        print("get testdatasets")


class TfRecordDatasetsLoader(BaseDatasetsLoader):
    def __init__(self, Origin_FIle_Path=[],Save_File_Path=[],name='TfRecordDatasetsLoader'):
        super(TfRecordDatasetsLoader,self).__init__(name=name)
        if len(Save_File_Path)==0:
                raise Exception("The Save_File_Path is not correct,Please Check it!",Save_File_Path)
        if len(Origin_FIle_Path)>1:
            print("Load Train and Test FIle\r\n")
            self.Origin_Train_FIle_Path = Origin_FIle_Path[0]
            self.Origin_Test_File_Path = Origin_FIle_Path[1]
            self.Save_Train_FIle_Path = Save_File_Path[0]
            self.Save_Test_FIle_Path = Save_File_Path[1]
            if not (os.path.isdir(self.Save_Train_FIle_Path)) or  not (os.path.isdir(self.Save_Test_FIle_Path)):
               os.makedirs(self.Save_Train_FIle_Path)
               os.makedirs(self.Save_Test_FIle_Path)
        elif len(Origin_FIle_Path)==1:
            print("Load Train  FIle\r\n")
            self.Origin_Train_FIle_Path = Origin_FIle_Path[0]
            self.Save_Train_FIle_Path = Save_File_Path[0]
            if not (os.path.isdir(self.Save_Train_FIle_Path)):
                os.makedirs(self.Save_Train_FIle_Path)
        elif len(Origin_FIle_Path)==0:
            raise Exception("The Origin_FIle_Path is not correct,Please Check it!",Origin_FIle_Path)

    def MakeTrainDatasets(self):
        '''rewrite your traindatasets make function,in this example, I rewrite this function for my own mvs-datasets'''
        def ParseTxtFunc(txt_file):
            img_name_pose_dict = dict() 
            pair_info = open(txt_file, "r")
            pair_file_context = pair_info.readline()
            line_count = 0
            while pair_file_context:
                pair_file_context = pair_info.readline()
                if len(pair_file_context)>12:
                    line_count = line_count+1
                    context_splits = pair_file_context.split(" ")
                    img_name = context_splits[0]
                    img_pose_R = np.array([ float(i) for i in context_splits[10:-3] ]).reshape((3,3))
                    img_pose_t = np.array([float(i) for i in context_splits[-3:]])
                    img_pose_quart = Rotation.from_matrix(img_pose_R).as_quat()
                    img_pose = np.concatenate([img_pose_t,img_pose_quart])
                    img_name_pose_dict[img_name]=img_pose
            return img_name_pose_dict
        #1.read image_label
        train_labels = []
        for home_dirs,dir_lists,file_lists in os.walk(self.Origin_Train_FIle_Path):
            #1.get labels from folder name
            if home_dirs ==self.Origin_Train_FIle_Path:
                if len(dir_lists)>=1:
                    for sub_dir in dir_lists:
                        if sub_dir.find("tfrecords")<0:
                            train_labels.append(sub_dir)
        #2.read image and pose
        train_data_dict = dict()
        for per_label in train_labels:
                #print("label_index-->",train_labels.index(per_label),per_label)
                train_data_file_path = os.path.join(self.Origin_Train_FIle_Path,per_label)
                for _,_,file_lists in os.walk(train_data_file_path):
                    for file in file_lists:
                        if file.find("_par")>0:
                            train_data_file_path = os.path.join(train_data_file_path,file)
                            train_data_img_pose_dict = ParseTxtFunc(train_data_file_path)
                            train_data_dict[train_labels.index(per_label)]=train_data_img_pose_dict
        #3.saved traindata in TFRecord
        tfrecord_file_name = os.path.join(self.Save_Train_FIle_Path,"TrainData.tfrecords")
        traindata_writer = tf.io.TFRecordWriter(tfrecord_file_name)
        for label in train_data_dict:
            for img_name in train_data_dict[label]:
                img_path = os.path.join(os.path.join(self.Origin_Train_FIle_Path,train_labels[label]),img_name)
                img =  tf.io.read_file(img_path)
                img_pose = train_data_dict[label][img_name]
                img_label = label#label.encode('utf-8')
                sample = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(img.numpy())])),
                                'img_pose': tf.train.Feature(float_list=tf.train.FloatList(value=img_pose)),
                                'img_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label]))
                            }
                        )
                    )
                traindata_writer.write(sample.SerializeToString())
        traindata_writer.close()
        print("Make Traindata Finished!\r\n")
    def ReadTrainDatasets(self,is_need_dataaug= True,is_center_crop=True,Center_Crop_Size=(128,128)):
        tfrecordfile_path = ' '
        #1.get tfrecordfile_path
        for home_list,dir_list,file_list in os.walk(self.Save_Train_FIle_Path):
            tfrecordfile_path = os.path.join(home_list,file_list[0])
        #2.parse tfrecordfile
        def parse_sample(example_proto):
            description = {
                'img':tf.io.FixedLenFeature([],tf.string),
                'img_pose':tf.io.FixedLenFeature([7],tf.float32),
                'img_label':tf.io.FixedLenFeature([1],tf.int64),
            }
            sample = tf.io.parse_single_example(example_proto,description)
            img = tf.image.decode_png(sample['img'],3)
            img =  tf.cast(img, dtype='float32')
            img = img/127.5-1.0
            #self.Data_Auguments(img,is_need_dataaug,is_center_crop,Center_Crop_Size)
            img_pose = sample['img_pose']
            img_label = sample['img_label'][0]#tf.one_hot(sample['img_label'][0],2,axis=-1)
            return img,img_pose,img_label
        train_datasets = tf.data.TFRecordDataset(
            tfrecordfile_path).map(parse_sample)
        return train_datasets


def Data_Auguments(traindata,is_need_dataaug=True,is_need_crop=True,croped_size=(128,128)):
        img,img_pose,img_label = traindata
        if is_need_crop:
            _,img_height,img_width,_ =img.shape
            offset_height = random.randint(croped_size[0],img_height-croped_size[0])
            offset_width = random.randint(croped_size[1],img_width-croped_size[1])
            img = tf.image.crop_to_bounding_box(img,offset_height,offset_width,croped_size[0],croped_size[1])
        else:
            img = tf.image.resize(img,croped_size)
        if is_need_dataaug:
            img = tf.image.random_brightness(img,0.2)
        
        return img,img_pose,img_label 




                
            
            


        



    



#########################################Test Module#######################################################
if __name__ =="__main__":
    m_tfrecorddataloader = TfRecordDatasetsLoader(["/home/swx/m_mode_datasets/temp_and_dino_datasets"],["/home/swx/m_mode_datasets/temp_and_dino_datasets/traindatasets_tfrecords"])
    #m_tfrecorddataloader.MakeTrainDatasets()
    traindataloader = m_tfrecorddataloader.ReadTrainDatasets().shuffle(300).batch(10)
    for i in traindataloader.take(10):
        i = Data_Auguments(i,True)
        print("img_shape--->",i[0].shape)
        # mp.imsave("traindata_{}.png".format(0),i[0].numpy())
        #print(i[1].numpy())
        print(i[2])