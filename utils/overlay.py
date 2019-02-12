#! /bin/env/ python3
'''
this file houses function(s) to manipulate video files to add overlays with
predicted probability values from some model
'''
# local imports
import color
import episode
import modelbuilder
# library imports
import numpy as np
from argparse import ArgumentParser
from progressbar import progressbar
from pathlib import Path
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def read_video(videopath):
    '''
    opens the video corresponding to supplied episode name, and returns an
    opencv2 VideoCapture object and a dictionary with metadata
    ---
        videopath: [required] the name of the videofile to open (expects mp4)

        return: tuple(VideoCapture, dict), where dict is a dictionary that
                looks like this: {'frames': f, 'width': w, 'height': h}
    '''
    videofile = Path(videopath)
    if videofile.suffix != '.mp4':
        videofile = videofile.joinpath('.mp4')
    if videofile.exists():
        cap = cv2.VideoCapture(str(videofile))
    else:
        raise FileNotFoundError


    metadata = dict(frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps = int(cap.get(cv2.CAP_PROP_FPS)),
                    fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)))

    return cap, metadata


def overlay_frame(frame, height=None, width=None, preds=None, index=None,
                  bufferlen=0, verbose=True):
    '''
    adds overlay to frame
    ---
        frame: numpy array of frame in the format returned by openCV2
        preds: numpy array of predictions of classes
        index: index of the prediction that corresponds to current frame
        bufferlen: length of how much data to show before and after current
                   frame [not yet implemented]
    '''
    if bufferlen != 0:
        raise NotImplementedError('bufferlen param not yet implemented')

    dpi = 100
    w = width / dpi
    h = height / dpi

    plt.clf()
    fig, ax = plt.subplots(1,1, figsize=(w, h))
    plt.axis('off')
    plt.imshow(frame)

    # before, after = preds[index-bufferlen:index], [index+1:index+bufferlen+1]
    this_pred = preds[index]
    bars = [plt.bar(1, this_pred[0]*height, .1*width)] # [int(width/2)]
    for c in range(1, preds.shape[-1]):
        bar = plt.bar(1, this_pred[c]*height, .1*width,
                      bottom=this_pred[c-1]*height)
        bars.append(bar)

    plt.legend((bar[0] for bar in bars), (c for c, _ in enumerate(bars)))

    # if verbose: plt.show()
    canvas = FigureCanvas(fig)
    canvas.draw()
    # plt.show()
    new_frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    size = fig.get_size_inches()*fig.dpi
    size = (int(x) for x in reversed(size))
    new_frame = new_frame.reshape(*size, 3)
    plt.close('all')

    # color.INFO('DEBUG', 'old: {}; shape of new: {}'.format(frame.shape,
    #                                                        new_frame.shape))

    # plt.plot(x,y,'k-', lw=2)
    # plt.plot(x[i],y[i],'or')

    return new_frame


def _overlay_video(cap, metadata, preds, writer, precision=2):
    '''
    the most basic video overlay method. needs a capture object, an array of
    predictions, and a precision level (with respect to samples per 0.96s
    chunks) in order to add an overlay to a video. method is internal because
    wrappers to initialize a capture object and make predictions are available.
    '''
    fps = metadata['fps']
    audioframe_dur = .96 / precision

    for i in progressbar(range(metadata['frames']), redirect_stdout=1):
        # color.INFO('DEBUG', 'processing frame {}'.format(i))

        time = i / fps
        predindex = time / audioframe_dur
        predindex = int(predindex)

        flag, frame = cap.read()
        new_frame = overlay_frame(frame, height=metadata['height'],
                                  width=metadata['width'], preds=preds,
                                  index=predindex)

        writer.write(new_frame)


def overlay_video(videopath, audiopath, model, precision=2):
    '''
    given filepaths for corresponding video and audio files, and model, uses
    the model to make predictions on audio, and adds an overlay to the video
    and saves it (at a different location) with prediction probabilities
    '''
    raise NotImplementedError

    # fig, ax = plt.subplots(1,1)
    # plt.ion()
    # plt.show()
    #
    # #Setup a dummy path
    # x = np.linspace(0,width,frames)
    # y = x/2. + 100*np.sin(2.*np.pi*x/1200)
    #
    # for i in range(frames):
    #     fig.clf()
    #     flag, frame = cap.read()
    #
    #     # print(flag, frame)
    #
    #     plt.imshow(frame)
    #     plt.plot(x,y,'k-', lw=2)
    #     plt.plot(x[i],y[i],'or')
    #     plt.pause(1e-100)
    #
    #     if cv2.waitKey(0) == 27:
    #         break


def overlay_episode(ep, model, precision=2):
    '''
    helper method that loads a video corresponding to the episode supplied,
    and uses the preds supplied to add overlay with those preds to the video
    '''
    videopath = Path('../video').joinpath(ep + '.mp4')
    out = Path('../video').joinpath(ep + '_preds-overlay.mp4' + '.mp4')
    audiopath = Path('../wav').joinpath(ep + '.wav')

    cap, metadata = read_video(str(videopath))
    writer = cv2.VideoWriter(str(out),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             metadata['fps'], (metadata['width'],
                                               metadata['height']))

    _, preds = episode.detect_in_episode(ep, model, precision=2, algorithms=[])
    preds = episode._binary_probs_to_multiclass(preds)

    _overlay_video(cap, metadata, preds, writer, precision)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-e', '--episode', type=str,
                            help='title (in standard format) of the episode to'
                                 ' add overlay to', default='friends-s03-e09')
    config = arg_parser.parse_args()

    model = modelbuilder.build_laugh_model()
    model.load_weights(filepath='../laughter/task:per-season-split-ckpt.hdf5')
    model = modelbuilder._compile_binary(model)

    overlay_episode(config.episode, model)
