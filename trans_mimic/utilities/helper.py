from shutil import copyfile
import datetime
import os
import ntpath
import torch


def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print(" Tensorboard session created: "+url)
    webbrowser.open_new(url)