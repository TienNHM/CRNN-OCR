# OCR USING CRNN AND CTC LOSS FUNCTION

## Dataset
- Synthetic Word Dataset
- Link: https://www.robots.ox.ac.uk/~vgg/data/text/
- Direct download: https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
- Dataset structure:

```
mnt
└───ramdisk
    └───max
        └───90kDICT32px
            ├───1
            │   ├───1
            │   ├───2
            │   ├───3
            |   .
            |   .
            │   └───7
            .
            .
            ├───3000
            │   ├───1
            │   ├───2
            │   ├───3
            │   .
            │   .
            │   └───7
            .
            .
            ├───3001
            ├───3002
            ├───3003
            .
            .
            └───4000
```

## Model
- Test set : Validate set = 90: 10
- Epochs = 10
- Summary:

![](img/summary.png)

## Results
- Loss:

![](img/10.png)

- Test accuracy:
```
Test accuracy (%):  68.74666666666667
```

- Test:

![](test/0-Classmates.png)
![](test/1-greater.png)
![](test/2-NUCLEAR.png)
![](test/3-reedy.png)
![](test/4-FLOWING.png)
![](test/5-Peens.png)
![](test/6-besots.png)
![](test/7-TELEPATHIC.png)
![](test/8-SWOOSHED.png)
![](test/9-Expend.png)