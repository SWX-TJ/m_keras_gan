import tensorflow as tf
import os
from scipy.spatial.transform import Rotation
import numpy as np

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
                train_data_file_path = os.path.join(self.Origin_Train_FIle_Path,per_label)
                #print("current traindata_path-->",train_data_file_path)
                for _,_,file_lists in os.walk(train_data_file_path):
                    for file in file_lists:
                        if file.find("_par")>0:
                            train_data_file_path = os.path.join(train_data_file_path,file)
                            train_data_img_pose_dict = ParseTxtFunc(train_data_file_path)
                            train_data_dict[per_label]=train_data_img_pose_dict
        #3.saved traindata in TFRecord
        tfrecord_file_name = os.path.join(self.Save_Train_FIle_Path,"TrainData.tfrecords")
        traindata_writer = tf.io.TFRecordWriter(tfrecord_file_name)
        for label in train_data_dict:
            for img_name in train_data_dict[label]:
                img_path = os.path.join(os.path.join(self.Origin_Train_FIle_Path,label),img_name)
                img =  tf.io.read_file(img_path)
                img_pose = train_data_dict[label][img_name]
                img_label = label.encode('utf-8')
                sample = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(img.numpy())])),
                                'img_pose': tf.train.Feature(float_list=tf.train.FloatList(value=img_pose)),
                                'img_label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_label]))
                            }
                        )
                    )
                traindata_writer.write(sample.SerializeToString())
        traindata_writer.close()
        print("Make Traindata Finished!\r\n")








                
            
            


        



    



#########################################Test Module#######################################################
if __name__ =="__main__":
    m_tfrecorddataloader = TfRecordDatasetsLoader(["/home/swx/m_mode_datasets/temp_and_dino_datasets"],["/home/swx/m_mode_datasets/temp_and_dino_datasets/traindatasets_tfrecords"])
    m_tfrecorddataloader.MakeTrainDatasets()