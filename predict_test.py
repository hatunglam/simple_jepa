import torch
import numpy as np

def predict_test(model):
    model.eval()
    print("-----------Start testing phase-----------")

    with torch.no_grad():
        test_img = torch.randn(5, 3, 100, 100)

        n_img = test_img.shape[0]

        for i in range(n_img):
            x = test_img[i].unsqueeze(0)

            target_aspect_ratio = np.random.uniform(model.target_aspect_ratio[0], model.target_aspect_ratio[1])
            target_scale = np.random.uniform(model.target_scale[0], model.target_scale[1])
            context_aspect_ratio = model.context_aspect_ratio
            context_scale = np.random.uniform(model.context_scale[0], model.context_scale[1])

            y_student, y_teacher = model(x,
                                         target_aspect_ratio, target_scale,
                                         context_aspect_ratio, context_scale)
            
            loss = model.criterion(y_student, y_teacher).item()

            print(f"\nSample {i+1}:")
            print(f"Prediction (student): {y_student.flatten()[:10]}...")  
            print(f"Ground truth (teacher): {y_teacher.flatten()[:10]}...")
            print(f"Loss = {loss:.6f}")