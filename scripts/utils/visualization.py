import captum
from captum.attr import visualization as viz
from .attribution import CaptumAlgorithm


class CaptumVisualizer():
    def __init__(self, imageloader):
        self.input_img = imageloader.process_images()[0]
        self.original_img = imageloader.process_images()[1]
    

    # Visualizes attribution for a given image by normalizing attribution values of the desired sign
    # (positive, negative, absolute value, or all) 
    # and displaying them using the desired mode in a matplotlib figure.
    def visualization_img_attr(
        self,
        algorithm: CaptumAlgorithm,
        target: int,
        method='heat_map',
        sign='absolute_value',
        plt_fig_axis=None,
        outlier_perc=2,
        cmap=None,
        alpha_overlay=0.5,
        show_colorbar=False,
        title=None,
        fig_size=(6, 6),
        use_pyplot=True):

        attribution = algorithm.attribution(self.input_img, target)
        attribution_img = attribution[0].cpu().permute(1,2,0).detach().numpy()

        figure, axis= viz.visualize_image_attr(
            attr=attribution_img,
            original_image=self.original_img,
            method=method,
            sign=sign,
            plt_fig_axis=plt_fig_axis,
            outlier_perc=outlier_perc,
            cmap=cmap,
            alpha_overlay=alpha_overlay,
            show_colorbar=show_colorbar,
            title=title,
            fig_size=fig_size,
            use_pyplot=use_pyplot
        )

        return figure, axis
    

    # Visualizes attribution using multiple visualization methods displayed in a 1 x k grid,
    # where k is the number of desired visualizations.
    def visualization_image_attr_multiple(
        self,
        algorithm: CaptumAlgorithm,
        target: int, 
        methods,
        signs,
        titles=None,
        fig_size=(8, 6),
        use_pyplot=True):

        attribution = algorithm.attribution(self.input_img, target)
        attribution_img = attribution[0].cpu().permute(1,2,0).detach().numpy()

        figure, axis = viz.visualize_image_attr_multiple(
            attr=attribution_img,
            original_image=self.original_img,
            methods=methods,
            signs=signs,
            titles=titles,
            fig_size=fig_size,
            use_pyplot=use_pyplot
            )

        return figure, axis