import numpy as np

import matplotlib.pyplot as plt

def filterBackground(img, tolerance=.05):
    backgroundColor = np.mean([np.median(img[i,:], axis=0) for i in range(img.shape[0])], axis=0)
    editedImg = img

    editedImg[np.sqrt(np.sum((img - backgroundColor)**2, axis=-1)) < tolerance*np.max(img),:] = 0
    return editedImg

def equalizeSpatialGradients(img, strength=.5, debug=False):
    horizontalGrad = np.array([np.median(img[:,i,:], axis=0) for i in range(img.shape[1])])
    verticalGrad = np.array([np.median(img[i,:,:], axis=0) for i in range(img.shape[0])])

    horizontalCorr = horizontalGrad - np.mean(horizontalGrad, axis=0)
    verticalCorr = verticalGrad - np.mean(verticalGrad, axis=0)

    corr = np.transpose([np.add.outer(horizontalCorr[:,i].T, verticalCorr[:,i].T) for i in range(3)])

    corrImg = img - corr*strength
    corrImg = (corrImg - np.min(corrImg))
    corrImg /= np.max(corrImg)

    # Now make sure the image has the same type and min/max as the input
    if img.dtype == np.uint8:
        corrImg *= 255
        corrImg = corrImg.astype(np.uint8)

    if debug:
        correctedHorizontalGrad = np.array([np.median(corrImg[:,i,:], axis=0) for i in range(img.shape[1])])
        correctedVerticalGrad = np.array([np.median(corrImg[i,:,:], axis=0) for i in range(img.shape[0])])

        fig, ax = plt.subplots(1, 2)
        labels = ['Red', 'Green', 'Blue']
        for i in range(3):
            ax[0].plot(horizontalGrad[:,i], label=labels[i], c=labels[i])
            ax[0].plot(correctedHorizontalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[0].set_title('Horizontal')
        ax[0].legend()
        for i in range(3):
            ax[1].plot(verticalGrad[:,i], label=labels[i], c=labels[i])
            ax[1].plot(correctedVerticalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[1].set_title('Vertical')
        ax[1].legend()
        fig.tight_layout()
        plt.show()

    return corrImg

def cropToContent(img, returnCorner=True):
    gray = np.mean(img, axis=-1)
    verticalBins = np.where(np.sum(gray, axis=0) > 0)
    if len(verticalBins[0]) > 0:
        leftBound, rightBound = verticalBins[0][0], verticalBins[0][-1]
    else:
        raise img

    horizontalBins = np.where(np.sum(gray, axis=1) > 0)
    if len(horizontalBins[0]) > 0:
        topBound, bottomBound = horizontalBins[0][0], horizontalBins[0][-1]
    else:
        raise img

    croppedImg = np.array(img)[topBound:bottomBound, leftBound:rightBound]
    return (croppedImg, (leftBound, topBound)) if returnCorner else croppedImg
