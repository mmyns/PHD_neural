import timm
NUM_FINETUNE_CLASSES = 1
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
model.eval()
