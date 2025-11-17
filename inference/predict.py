def predict(model, img):
    model.eval()
    return model(img)
