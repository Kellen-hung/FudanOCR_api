# main.py
import torch
import torch.nn as nn
from config import config
from util import get_data_package, get_alphabet, get_radical_alphabet, saver, must_in_screen, converter
from model_loader import build_main_model, build_clip_model, build_optimizer_scheduler
from feature_extractor import extract_text_features
from train import Trainer

def main():

    saver()
    # must_in_screen()

    alphabet = get_alphabet()
    radical_alphabet = get_radical_alphabet()
    
    print(len(radical_alphabet))

    model = build_main_model()
    
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_dis = nn.MSELoss().cuda()

    clip_model = build_clip_model(len(radical_alphabet))
    text_features = extract_text_features(clip_model, radical_alphabet, config['alpha_path'])

    optimizer, scheduler = build_optimizer_scheduler(model)

    train_loader, validation_loader = get_data_package()
    
    # 建立 Trainer
    print('\033[32m[info] Building trainer... \033[0m')
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        criterion_dis=criterion_dis,
        train_loader=train_loader,
        validation_loader=validation_loader,
        text_features=text_features,
        alphabet=alphabet,
        exp_name=config['exp_name']
    )
    if not config['test']:
        
        print(f"\033[33m[info] ------------------------- Mode: train and validation! -------------------------\033[0m")
        
        # train and validation
        for epoch in range(1, config['epoch']+1):
            # 備份
            torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))
            
            for iteration, data in enumerate(train_loader):
                image, label, index = data
                image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

                length, text_input, text_gt, _ = converter(label)
                trainer.train_one_step(epoch, iteration+1, image, length, text_input, text_gt)

            trainer.validation(epoch)
            scheduler.step()

    else:
        print(f"\033[33m[info] Mode: test! \033[0m")
        if config['resume_model'] == '':
            print(f"\033[31m[Error] The \'resume model\' in config.py can not be empty!!! \033[0m")
            return
        trainer.validation()

if __name__ == "__main__":
    main()
