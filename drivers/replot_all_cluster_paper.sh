#! /bin/bash
#
# Usage: when tweaking e.g., earhart.helpers.get_autorotation_dataframe
# selection criteria... remake the

cd cluster_rotation;
python plot_autorotation.py;

cd ../
python plot_rotation_X_RUWE.py

python plot_ngc2516_corehalo_3panel.py

python plot_physical_X_rotation.py

python plot_skypositions_x_rotn.py

python plot_full_kinematics_X_rotation.py
