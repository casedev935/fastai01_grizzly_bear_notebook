# fastai01_grizzly_bear_notebook

[![Jupyter](https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg)](https://jupyter.org/) 

## Introduction

This is a basic training model for classifying bear types. 

The code was made through the FASTAI course, Lesson #1 [https://course.fast.ai/Lessons/lesson1.html] plus some functions suggested in their forum. 

## Requirements

- icrawler (ImageDownloader)
- icrawler.builtin (GoogleImageCrawler)
- icrawler.builtin.google (GoogleFeeder, GoogleParser)
- fastai.vision.all

## Main Functions

- search_images(term, max_images=30, folder_name=".")
- DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)
- vision_learner(DataBlock, Pre-Trained Model, metrics=error_rate)

## Pretrained Model

- RESNET-18
*Used plot_confusion_matrix() to see confusion matrix
