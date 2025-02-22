from manim import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

class Arrow3D(VMobject):
    def __init__(
        self, 
        start, 
        end, 
        color=WHITE, 
        stroke_width=2, 
        opacity=1, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.start = np.array(start)
        self.end = np.array(end)
        # Create the main line using Line3D.
        line = Line3D(self.start, self.end, color=color, stroke_width=stroke_width)
        # Determine tip length relative to the line's length.
        tip_length = 0.15 * np.linalg.norm(self.end - self.start)
        # Create an arrow tip as a Cone.
        tip = Cone(
            base_radius=0.05,
            height=tip_length,
            direction=(self.end - self.start) / np.linalg.norm(self.end - self.start),
            fill_color=color,
            fill_opacity=opacity,
            stroke_width=stroke_width,
        )
        # Position the tip at the end of the line.
        tip.shift(self.end - tip_length * (self.end - self.start) / np.linalg.norm(self.end - self.start))
        self.add(line, tip)
        self.set_opacity(opacity)

    def get_start(self):
        # Override to return our stored start position.
        return self.start

    def get_end(self):
        # Override to return our stored end position.
        return self.end

class IrisPCA(ThreeDScene):
    def construct(self):
        # Set up 3D axes and camera orientation.
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Load the Iris dataset and select the first 3 features:
        # (sepal length, sepal width, petal length)
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target  # 0, 1, 2 for the three classes

        # Center the data so it appears in the middle of the coordinate system.
        X = X - np.mean(X, axis=0)
        
        # Define distinct colors for each class.
        class_colors = [RED, GREEN, BLUE]
        
        # Create Dot3D objects for each data point with the class color.
        dot_list = []
        for i, point in enumerate(X):
            dot = Dot3D(point=point, color=class_colors[y[i]], radius=0.05)
            dot_list.append(dot)
        dots = VGroup(*dot_list)
        
        # Compute PCA on the entire dataset (from 3D to 2D).
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        # Embed the 2D PCA result in 3D (z=0 for all points).
        X_pca_3d = np.column_stack((X_pca, np.zeros(len(X))))
        
        # For each class, compute the 1D PCA (max variance direction) and draw an arrow.
        arrows = VGroup()
        for class_label in np.unique(y):
            X_class = X[y == class_label]
            mean_class = X_class.mean(axis=0)
            pca_class = PCA(n_components=1)
            pca_class.fit(X_class)
            # Principal component direction.
            direction = pca_class.components_[0]
            # Scale the vector by the standard deviation along that direction.
            scale = np.std(X_class @ direction)
            vector = direction * scale
            start_arrow = mean_class - vector / 2
            end_arrow = mean_class + vector / 2
            arrow = Arrow3D(
                start=start_arrow,
                end=end_arrow,
                color=class_colors[int(class_label)],
                stroke_width=2,
                opacity=0.3
            )
            arrows.add(arrow)
        
        # Display the initial scene with axes, data points, and variance arrows.
        self.play(Create(axes), FadeIn(dots), FadeIn(arrows))
        self.wait(1)
        
        # Animate the transition of data points from their original 3D positions
        # to the PCA-projected 2D positions.
        dot_anims = [dot.animate.move_to(new_pos) 
                     for dot, new_pos in zip(dot_list, X_pca_3d)]
        
        # Animate the transformation of the variance arrows.
        arrow_anims = []
        for arrow in arrows:
            old_start = arrow.get_start()
            old_end = arrow.get_end()
            # Apply the same PCA transformation to the arrow endpoints.
            new_start_2d = pca.transform(np.array([old_start]))[0]
            new_end_2d = pca.transform(np.array([old_end]))[0]
            new_start_3d = np.array([new_start_2d[0], new_start_2d[1], 0])
            new_end_3d = np.array([new_end_2d[0], new_end_2d[1], 0])
            arrow_anims.append(
                arrow.animate.become(
                    Arrow3D(
                        start=new_start_3d,
                        end=new_end_3d,
                        color=arrow.get_color(),
                        stroke_width=2,
                        opacity=0.3
                    )
                )
            )
        
        self.play(
            AnimationGroup(
                *dot_anims,
                *arrow_anims,
                lag_ratio=0,
                run_time=3
            )
        )
        self.wait(2)
