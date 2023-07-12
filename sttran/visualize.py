from PIL import Image, ImageDraw
import pydot
import cv2
import numpy as np
import os
from tqdm import tqdm


def make_scene_graphs(annotation, image_folder, show=False, save=True, save_folder='graphs'):
    if show:
        raise NotImplementedError("Have not implemented showing graphs inline yet")
    if save:
        os.makedirs(save_folder, exist_ok=True)
    # with open(annotation_file, 'rb') as f:
    #     annotation = pickle.load(f)

    for image_name in tqdm(annotation.keys()):
        frame_anno = annotation[image_name]
        image_last = image_name.split('/')[-1]
        image_number = image_last.split('.')[0]
        image_file = os.path.join(image_folder, image_last)
        img = Image.open(image_file)
        draw = ImageDraw.Draw(img)
        draw.text((10,10), image_name[10:], fill='white')
        
        for i in frame_anno:
            if i['visible']:
                draw.rectangle([i['bbox'][0], i['bbox'][1], i['bbox'][0]+i['bbox'][2], i['bbox'][1]+i['bbox'][3]], None, 'red')
                draw.text((i['bbox'][0], i['bbox'][1]), i['class'], fill='white')
        img.save('{}/{}{}.png'.format(save_folder, image_number, 'F'))

        #contact
        G = pydot.Dot(graph_type='digraph')
        cluster_contact=pydot.Cluster('contact',label='contact')
        G.add_subgraph(cluster_contact)
        person_node = pydot.Node('person', style='filled', fillcolor='royalblue')
        cluster_contact.add_node(person_node)
        for i in frame_anno:
            if i['visible']:
                object_node = pydot.Node(i['class'], style='filled', fillcolor='bisque')
                cluster_contact.add_node(object_node)
                for j in i['contacting_relationship']:
                    cluster_contact.add_edge(pydot.Edge(person_node, object_node, label=j))
        G.write_png('{}/{}{}.png'.format(save_folder, image_number, 'C'))

        #attention
        G = pydot.Dot(graph_type='digraph')
        cluster_contact=pydot.Cluster('attention',label='attention')
        G.add_subgraph(cluster_contact)
        person_node = pydot.Node('person', style='filled', fillcolor='royalblue')
        cluster_contact.add_node(person_node)
        for i in frame_anno:
            if i['visible']:
                object_node = pydot.Node(i['class'], style='filled', fillcolor='bisque')
                cluster_contact.add_node(object_node)
                for j in i['attention_relationship']:
                    cluster_contact.add_edge(pydot.Edge(person_node, object_node, label=j))
        G.write_png('{}/{}{}.png'.format(save_folder,  image_number, 'A'))

        #spatial
        G = pydot.Dot(graph_type='digraph')
        cluster_contact=pydot.Cluster('attention',label='spatial')
        G.add_subgraph(cluster_contact)
        person_node = pydot.Node('person', style='filled', fillcolor='royalblue')
        cluster_contact.add_node(person_node)
        for i in frame_anno:
            if i['visible']:
                object_node = pydot.Node(i['class'], style='filled', fillcolor='bisque')
                cluster_contact.add_node(object_node)
                for j in i['spatial_relationship']:
                    cluster_contact.add_edge(pydot.Edge(object_node, person_node, label=j))
        G.write_png('{}/{}{}.png'.format(save_folder, image_number, 'S'))

    #generate the video
    fps = 1
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter('{}/result.avi'.format(save_folder), fourcc, fps, (1200,900))
    for image_name in annotation.keys():
        path = image_name[10:-4]
        img1 = cv2.imread('{}/{}A.png'.format(save_folder, path))
        img1 = cv2.resize(img1, (600, 300))
        img2 = cv2.imread('{}/{}S.png'.format(save_folder, path))
        img2 = cv2.resize(img2, (600, 300))
        img3 = cv2.imread('{}/{}C.png'.format(save_folder, path))
        img3 = cv2.resize(img3, (600, 300))
        img4 = cv2.imread('{}/{}F.png'.format(save_folder, path))
        img4 = cv2.resize(img4, (600, 900))
        vtitch = np.vstack((img1, img2, img3))
        htitch = np.hstack((img4, vtitch))
        videoWriter.write(htitch)
    videoWriter.release()