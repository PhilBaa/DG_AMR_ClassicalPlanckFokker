import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

theme = pv.global_theme
theme.font.label_size = 50
theme.font.title_size = 50

pv.global_theme.colorbar_horizontal.position_x = 0.2

pl = pv.Plotter(notebook=False, off_screen=True, shape = (1, 2), theme=theme)
# read the data
grid = pv.read('build/solutions/solution-000.vtk')
surface = grid.separate_cells().extract_surface(nonlinear_subdivision=4)
edges = surface.extract_feature_edges(0)

pl.subplot(0, 0)
c_plot = pl.add_mesh(grid,
                    scalars='U',
                    )

pl.subplot(0, 1)
line_plot = pl.add_mesh(edges, color='black',
                        line_width=1.0,
                        render_lines_as_tubes=True,
                        ) 

# plot the data with an automatically created Plotter
for i in range(2):
    pl.subplot(0,i)
    pl.window_size = [3008, 2000]
    pl.set_scale(yscale=0.3)
    pl.camera_position = 'xy'



# Create a plotter object and set the scalars to the Z height

# Open a gif
pl.open_movie("wigner.mp4")

pts = grid.points.copy()

seismic = plt.get_cmap('seismic')
jump_strength = 5
opacity = jump_strength*[1] + [0] + jump_strength*[1]

pl.subplot(0, 0)
labels = dict(xtitle='x', ytitle='p')
pl.show_grid(**labels, show_zaxis=False, n_xlabels=3, n_ylabels=3)


steps = [10*i for i in range(828)]
for step in steps:
    pl.subplot(0, 0)
    pl.remove_actor(c_plot, render=False)
    new = pv.read(f'build/solutions/solution-{step:03d}.vtk')
    upper = 1.2* np.max(new['U'])
    lower = -upper
    c_plot = pl.add_mesh(new, scalars='U', render=False, cmap=seismic, clim=[lower, upper], opacity=opacity)


    pl.subplot(0, 1)
    pl.remove_actor(line_plot, render=False)
    new_surface = new.separate_cells().extract_surface(nonlinear_subdivision=4)
    new_edges = new_surface.extract_feature_edges(0)
    line_plot = pl.add_mesh(new_edges, color='black',
                            line_width=1.5,
                            render=False)
    # Write a frame. This triggers a render.
    pl.write_frame()
    print(f"Frame {int(step/10)}/{len(steps)} written", end='\r')

# Closes and finalizes movie
pl.close()



