"""Module containing scripts for creating and reproducing the dataset synthesised from a 3D model of a chess set.

Notice that the Blender script for actually creating the dataset is located in ``scripts/synthesize_data.py`` and not the `chesscog` package itself because it uses ``bpy`` and Blender's bundled Python interpreter.
Thus the dependencies are not in line with `chesscog` itself.

It is recommended to use the :mod:`~chesscog.data_synthesis.download_dataset` script to download the rendered dataset and then to split it into train/val/test using the :mod:`~chesscog.data_synthesis.split_dataset` script.
"""
