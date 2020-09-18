# Screencast Tutorial Video Understanding
PyTorch code for the paper "Screencast Tutorial Video Understanding" [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Screencast_Tutorial_Video_Understanding_CVPR_2020_paper.pdf)], which is accepted by CVPR 2020. 



## Introduction
Screencast tutorials are videos created by people to teach how to use software applications or demonstrate procedures for accomplishing tasks. It is very popular for both novice and experienced users to learn new skills, compared to other tutorial media such as text, because of the visual guidance and the ease of understanding. In this paper, we propose visual understanding of screencast tutorials as a new research problem to the computer vision community. We collect a new dataset of Adobe Photoshop video tutorials and annotate it with both low-level and high-level semantic labels. We introduce a bottom-up pipeline to understand Photoshop video tutorials. We leverage state-of-the-art object detection algorithms with domain specific visual cues to detect important events in a video tutorial and segment it into clips according to the detected events. We propose a visual cue reasoning algorithm for two high-level tasks: video retrieval and video captioning. We conduct extensive evaluations of the proposed pipeline. Experimental results show that it is effective in terms of understanding video tutorials. We believe our work will serves as a starting point for future research on this important application domain of video understanding.


![pipeline](/fig/pipeline.png)


## Text-Tutorial Clip Retrieval
Code, extracted feature, pretrained model and doc for text-to-tutorial clip retrieval task are in the "matching_code/" folder. 


## Tutorial Clip Captioning
Code, extracted feature, pretrained model and doc for tutorial clip captioning task are in the "captioning_code/" folder.


## Source Data
The source videos as well as annotations can be downloaded from: https://drive.google.com/drive/folders/1osWW6dnsnvlWNseOtivIdhdpVct1r38x?usp=sharing, where "video_clips.zip" include video clips after the temporal segmentation and "whole_video.zip" includes the original complete tutorials.


## Reference
If you found this code useful, please cite the following paper:


If you found this code useful, please cite the following paper:

    @inproceedings{li2020pstuts,
      title={Screencast Tutorial Video Understanding},
      author={Li, Kunpeng and Fang, Chen and Wang, Zhaowen and Kim, Seokhwan and Jin, Hailin and Fu, Yun},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2020}
    }


## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

