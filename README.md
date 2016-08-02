# calibration
Multi-Camera Self-Calibration

# Input file format.
Comments start with #.

Format data file.
```
#resolutions
<1st_camera_width>; <1st_camera_height>; <2nd_camera_width>; <2nd_camera_height>; ... ; <kth_camera_with>; <kth_camera_height>

#truth
<1st_frame_number>; <frame_time>; <1st_camera_point_x>; <1st_camera_point_y>; <2nd_camera_point_x>; <2nd_camera_point_y>; ... ; <kth_camera_point_x>; <kth_camera_point_y>
<2nd_frame_number>; <frame_time>; <1st_camera_point_x>; <1st_camera_point_y>; <2nd_camera_point_x>; <2nd_camera_point_y>; ... ; <kth_camera_point_x>; <kth_camera_point_y>
...
<mth_frame_number>; <frame_time>; <1st_camera_point_x>; <1st_camera_point_y>; <2nd_camera_point_x>; <2nd_camera_point_y>; ... ; <kth_camera_point_x>; <kth_camera_point_y>

# data
<mth+1_frame_number>; <frame_time>; <1st_camera_point_x>; <1st_camera_point_y>; <2nd_camera_point_x>; <2nd_camera_point_y>; ... ; <kth_camera_point_x>; <kth_camera_point_y>
...
<nth_frame_number>; <frame_time>; <1st_camera_point_x>; <1st_camera_point_y>; <2nd_camera_point_x>; <2nd_camera_point_y>; ... ; <kth_camera_point_x>; <kth_camera_point_y>
```

Format of file with ground truth.
```
<1st_frame_point_x> <1st_frame_point_y> <1st_frame_point_z>
<2nd_frame_point_x> <2nd_frame_point_y> <2nd_frame_point_z>
...
<mth_frame_point_x> <mth_frame_point_y> <mth_frame_point_z>
```

The first data line of the data file contains list of camera resolutions. The subsequent lines contain positions of the pointer in image coordinates for respective cameras.

The frame numbers and times are irrelevant for the main program, however they are useful for debugging purposes. They need to contain some fake values at least.

The first `m` points has special meaning, they should have given corresponding real world positions in the truth file to find real world frame of reference.

The data file can contain a `nan` value instead of numeric value, if data is missing for given position. E.g.

```
4535; 18.023216323653923; 41; 4; -121; 35; nan; nan; nan; nan; -70; 352
```
