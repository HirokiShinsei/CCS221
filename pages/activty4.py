import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def _plt_basic_object_(points):
    tri = Delaunay(points).convex_hull
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    s = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                        triangles=tri,
                        shade=True, cmap=cm.rainbow, lw=0.5)
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    st.pyplot(fig)


def _cube_(bottom_lower=(0, 0, 0), side_length=5):
    bottom_lower = np.array(bottom_lower)
    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [side_length, 0, 0],
        bottom_lower + [0, 0, side_length],
        bottom_lower + [0, side_length, side_length],
        bottom_lower + [side_length, side_length, side_length],
        bottom_lower + [side_length, 0, side_length],
        bottom_lower,
    ])
    return points


def _triangular_prism_(bottom_lower=(0, 0, 0), side_length=5, height=5):
    bottom_lower = np.array(bottom_lower)
    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [0, 0, height],
        bottom_lower + [0, side_length, height],
        bottom_lower + [side_length, side_length, height],
        bottom_lower,
    ])
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    points = np.dot(points, rotation_matrix.T)
    return points


def _pyramid_(bottom_lower=(0, 0, 0), side_length=5, height=5):
    bottom_lower = np.array(bottom_lower)
    base_points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [side_length, 0, 0],
    ])
    top_point = bottom_lower + [side_length / 2, side_length / 2, height]
    points = np.vstack([base_points, top_point])
    return points


def _rectangular_prism_(bottom_lower=(0, 0, 0), length=5, width=5, height=5):

    bottom_lower = np.array(bottom_lower)

    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, width, 0],
        bottom_lower + [length, width, 0],
        bottom_lower + [length, 0, 0],
        bottom_lower + [0, 0, height],
        bottom_lower + [0, width, height],
        bottom_lower + [length, width, height],
        bottom_lower + [length, 0, height],
        bottom_lower,
    ])

    return points


def translate_obj(points, amount):
    return tf.add(points, amount)

#defining each rotational matrix
def x_rotation_matrix(theta):
    theta = np.radians(theta)
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def y_rotation_matrix(theta):
    theta = np.radians(theta)
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def z_rotation_matrix(theta):
    theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

#defining the main rotation transformation function
def rotate_object(points, rot_x, rot_y, rot_z):
    points = tf.matmul(points, x_rotation_matrix(rot_x))
    points = tf.matmul(points, y_rotation_matrix(rot_y))
    points = tf.matmul(points, z_rotation_matrix(rot_z))
    return points


init_cube = _cube_(side_length=5)
points_cube = tf.constant(init_cube, dtype=tf.float32)

init_triangular_prism = _triangular_prism_(side_length=5, height=5)
points = tf.constant(init_triangular_prism, dtype=tf.float32)

init_pyramid = _pyramid_(side_length=5, height=5)
points2 = tf.constant(init_pyramid, dtype=tf.float32)

init_rectangular_prism = _rectangular_prism_(length=5, width=3, height=4)
points3 = tf.constant(init_rectangular_prism, dtype=tf.float32)

translation_amount = tf.constant([1, 2, 3], dtype=tf.float32)

translated_object = translate_obj(points, translation_amount)
translated_object2 = translate_obj(points2, translation_amount)
translated_object3 = translate_obj(points_cube, translation_amount)
translated_object4 = translate_obj(points3, translation_amount)

with tf.compat.v1.Session() as session:
    translated_triangular_prism = session.run(translated_object)

with tf.compat.v1.Session() as session:
    translated_pyramid = session.run(translated_object2)

with tf.compat.v1.Session() as session:
    translated_cube = session.run(translated_object3)

with tf.compat.v1.Session() as session:
    translated_rectangular_prism = session.run(translated_object4)

st.title("3D Object Translation")
st.header("Original and Translated Objects")
st.write("Use the sliders to translate the objects in the x, y, and z directions.")

# creates sliders to adjust translation amount
translation_x = st.slider("X", -5.0, 5.0, 1.0, 0.1)
translation_y = st.slider("Y", -5.0, 5.0, 2.0, 0.1)
translation_z = st.slider("Z", -5.0, 5.0, 3.0, 0.1)

# creates sliders for rotation angles
rot_x = st.slider("Rotate around X-axis", -180, 180, 0, 1)
rot_y = st.slider("Rotate around Y-axis", -180, 180, 0, 1)
rot_z = st.slider("Rotate around Z-axis", -180, 180, 0, 1)

# create translated objects
translation_amount = tf.constant([translation_x, translation_y, translation_z], dtype=tf.float32)
translated_triangular_prism = translate_obj(points, translation_amount)
translated_pyramid = translate_obj(points2, translation_amount)
translated_cube = translate_obj(points_cube, translation_amount)
translated_rectangular_prism = translate_obj(points3, translation_amount)

# run TensorFlow session to get translated objects
with tf.compat.v1.Session() as session:
    translated_triangular_prism = session.run(translated_triangular_prism)
    translated_pyramid = session.run(translated_pyramid)
    translated_cube = session.run(translated_cube)
    translated_rectangular_prism = session.run(translated_rectangular_prism)
    
rotated_triangular_prism = rotate_object(translated_triangular_prism, rot_x, rot_y, rot_z)
rotated_pyramid = rotate_object(translated_pyramid, rot_x, rot_y, rot_z)
rotated_cube = rotate_object(translated_cube, rot_x, rot_y, rot_z)
rotated_rectangular_prism = rotate_object(translated_rectangular_prism, rot_x, rot_y, rot_z)

# plot original and transformed objects
st.write("Triangular Prism")
_plt_basic_object_(init_triangular_prism)
_plt_basic_object_(translated_triangular_prism)
_plt_basic_object_(rotated_triangular_prism)

st.write("Pyramid")
_plt_basic_object_(init_pyramid)
_plt_basic_object_(translated_pyramid)
_plt_basic_object_(rotated_pyramid)

st.write("Cube")
_plt_basic_object_(init_cube)
_plt_basic_object_(translated_cube)
_plt_basic_object_(rotated_cube)

st.write("Rectangular Prism")
_plt_basic_object_(init_rectangular_prism)
_plt_basic_object_(translated_rectangular_prism)
_plt_basic_object_(rotated_rectangular_prism)