import torch
import numpy as np

def predict_test(model):
    model.eval()
    print("-----------Start RGBD Testing Phase-----------")

    with torch.no_grad():
        test_rgb = torch.randn(5, 3, 100, 100)   
        test_dep = torch.randn(5, 1, 100, 100)   

        n_img = test_rgb.shape[0]

        for i in range(n_img):
            x_rgb = test_rgb[i].unsqueeze(0)  # --> (1, 3, 100, 100)
            x_dep = test_dep[i].unsqueeze(0)  # --> (1, 1, 100, 100)

            target_aspect_ratio = np.random.uniform(model.target_aspect_ratio[0], model.target_aspect_ratio[1])
            target_scale = np.random.uniform(model.target_scale[0], model.target_scale[1])
            context_aspect_ratio = model.context_aspect_ratio
            context_scale = np.random.uniform(model.context_scale[0], model.context_scale[1])

            y_student, y_teacher = model(
                x_rgb, x_dep,
                target_aspect_ratio, target_scale,
                context_aspect_ratio, context_scale
            )

            loss = model.criterion(y_student, y_teacher).item()

            print(f"\nSample {i+1}:")
            print(f"Prediction (student): {y_student.flatten()[:10]}")
            print(f"Ground truth (teacher): {y_teacher.flatten()[:10]}")
            print(f"Loss = {loss:.6f}")