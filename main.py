from __future__ import print_function

import sys
import psutil
import os

import numpy as np
import gc
import telebot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from io import BytesIO
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
import torchvision.models as models

token = '5069703088:AAEA-gfnn-tn9yy6i616eNtiV4sEUJa3x4E'
bot = telebot.TeleBot(token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pool_images = dict()

imsize = 512 #if torch.cuda.is_available() else 128  # use small size if no gpu

resizer = transforms.Resize(imsize)
loader = transforms.ToTensor() # transform it into a torch tensor

def download_images(chat_id, ind): # Скачаем полученную нами картинку
    file_info = bot.get_file(pool_images[chat_id][ind])
    downloaded_file = bot.download_file(file_info.file_path)
    return downloaded_file

def bytes_to_pil(img): #Переведём картинку из типа bytes в PIL, съедобный для PyTorch
    stream = BytesIO(img)
    image = Image.open(stream).convert("RGB")
    stream.close()
    image = resizer(image)
    size = [i for i in image.size]
    return image, size

def image_loader(image_name): #Приведем PIL картинку к тензору
    image = image_name
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()
torch.save(cnn[:11], 'my_new_model.pth')
new_loaded_model = torch.load('my_new_model.pth')

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    gc.collect()
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 1 == 0:
                pid = os.getpid()
                python_process = psutil.Process(pid)
                memoryUse = python_process.memory_info()[0] / 2. ** 30  # memory use in GB...I think
                print('memory use:', memoryUse)
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
        gc.collect()
    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    im = Image.fromarray(input_tensor)
    # im.save(filename) # Если потребуется сохранить на диск - раскомментировать
    return im

def compare_two_pics(cont_pic, style_pic):
    image_cont, size_cont = cont_pic
    image_style, size_style = style_pic # Вытащили размеры и картинки
    smaller_side = np.argmin(size_style)  # Получили индекс наименьшей стороны изображения (по ней и будем равнять)
    ratio_c = (size_cont[1] / size_cont[0]) # Посчитали соотношение сторон
    ratio_s = (size_style[1] / size_style[0])

    if ratio_s == 1: # Если соотношение сторон стиля - квадрат, напрямую меняем размеры
        if ratio_c<ratio_s:
            size_style[1] = round(size_style[0] * ratio_c)
        else:
            size_style[0] = round(size_style[1] / ratio_c)
    elif ratio_s < 1 and ratio_c < 1 and ratio_c < ratio_s: # превышение контента по любой из сторон, при одинаковой ориентации
        size_style[1] = round(size_style[1] * ratio_c)
    elif ratio_s > 1 and ratio_c > 1 and ratio_c > ratio_s:
        size_style[0] = round(size_style[0] / ratio_c)
    elif ratio_c < ratio_s: # Если пропорции отличны от квадрата, то сравниваем их и получаем значения бОльшей стороны
        biggest_side = round(size_style[smaller_side] * ratio_c)
        size_style[np.argmax(size_style)] = biggest_side # Меняем бОльшую исходную сторону на полученную
    elif ratio_c > ratio_s:
        biggest_side = round(size_style[smaller_side] / ratio_c)
        size_style[np.argmax(size_style)] = biggest_side
        # Если пропроции одинаковы, то ничего не меняем
    if ratio_c <= 1:
        image_crop = transforms.RandomCrop(size=(min(size_style), max(size_style)))
        image_resize = transforms.Resize((min(size_cont), max(size_cont)))
    elif ratio_c > 1:
        image_crop = transforms.RandomCrop(size=(max(size_style), min(size_style)))
        image_resize = transforms.Resize((max(size_cont), min(size_cont)))

    image_style = image_crop(image_style)
    image_style = image_resize(image_style)
    return image_style

def get_image(message_chat_id, photo_id): # Работа с полученными изображениями
    global pool_images
    if message_chat_id in pool_images:
        if len(pool_images[message_chat_id]) == 2:
            del pool_images[message_chat_id] # Если две фотографии с чата уже поступили, то чистим пул
            pool_images[message_chat_id] = [photo_id] # После очистки пула создаём новую пару: чат - фото_контент
            return None
        else: # Если, первая фотография с контентом была уже получена, то добавим фото_стиль и выполним работу
            pool_images[message_chat_id].append(photo_id)
            bot.send_message(message_chat_id, "Я получил пример стиля, теперь осталось немножко подожать =) Обычно это занимает не больше 15 минут")
            content_img_bytes = download_images(chat_id=message_chat_id, ind=0) # Скачаем изображения
            style_img_bytes = download_images(chat_id=message_chat_id, ind=1)

            cont_PIL = bytes_to_pil(content_img_bytes) # Приведём к PIL и ресайз
            style_PIL = bytes_to_pil(style_img_bytes)

            new_cont_PIL, cont_size = cont_PIL # Уберём индексы
            new_st_PIL = compare_two_pics(cont_PIL, style_PIL) # подгоним изображение по размеру под контент

            cont_img = image_loader(new_cont_PIL)
            st_img = image_loader(new_st_PIL)
            input_img = cont_img.clone()
            output = run_style_transfer(new_loaded_model, cnn_normalization_mean, cnn_normalization_std,
                                        cont_img, st_img, input_img, num_steps=39)
            return output
    else:
        pool_images[message_chat_id] = [photo_id] # Если Пользователь новый, то создаём пару чат - фото_контент
        return None


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,"Привет! Я бот, созданный для переноса стиля с одного изображения на другой")
    bot.send_message(message.chat.id,"Пришли мне два изображения и я выдам их комплиментированный вариант")

@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.chat.id, 'Йо!')
    elif message.text.lower() == 'пока':
        bot.send_message(message.chat.id, 'Пока :)')

@bot.message_handler(content_types = ['photo'])
def get_image_message(message):
    photo_id = message.photo[-1].file_id #сохраняем ID картинки
    outp = get_image(message_chat_id=message.chat.id, photo_id=photo_id)
    if outp != None:
        outp = save_image_tensor2pillow(outp, 'out_image.jpg')
        bot.send_message(message.chat.id, "А вот и результат!")
        bot.send_photo(message.chat.id, outp)

        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory use:', memoryUse, 'память после выполнения работы')

    if len(pool_images[message.chat.id]) == 1:
        bot.send_message(message.chat.id, "Отлично, изображение с контентом получено, теперь жду изображение со стилем")

bot.infinity_polling()
