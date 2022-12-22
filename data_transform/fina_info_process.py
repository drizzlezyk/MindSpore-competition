import json


with open("./groups_members_info.json", 'r', encoding='UTF-8') as load_f:
    patri_list = json.load(load_f)

print(len(patri_list))

with open("competition_20221111/competition/groups.json", 'r', encoding='UTF-8') as load_f:
    group_list = json.load(load_f)

with open("competition_20221111/competition/score_new.json", 'r', encoding='UTF-8') as load_f:
    score_list = json.load(load_f)

with open("competition_20221111/competition/group_score_new.json", 'r', encoding='UTF-8') as load_f:
    group_score = json.load(load_f)

with open("competition_20221111/competition/project.json", 'r', encoding='UTF-8') as load_f:
    project_list = json.load(load_f)


final_info_list = [[] * 1 for _ in range(3)]

for parti in patri_list:
    group_id = parti['group_id']
    temp_dict = dict()
    for group in group_list:
        if group['id'] == group_id:
            for parti in patri_list:
                if group['leader_id'] == parti['user_id']:
                    temp_dict = dict()
                    temp_dict['team_name'] = group['name']
                    temp_dict['phone'] = parti['phone']
                    temp_dict['email'] = parti['email']

                    for project in project_list:
                        if group['relate_project_id'] == project['id']:
                            temp_dict['leader'] = parti['username']
                            temp_dict['repo'] = parti['username'] + '/' + project['name']
                            competition_id = int(group['relate_competition_id']) - 1

                            if group['period'] == '2':
                                final_info_list[competition_id].append(temp_dict)
                            break
print(final_info_list)

new_project_dict = dict()
new_project_dict['info'] = final_info_list[2]
out_base = './database/image_classification/'
user_info_path = out_base + 'final-info.json'
with open(user_info_path, 'w') as f:
    f.write(json.dumps(new_project_dict, indent=1, ensure_ascii=False))

new_project_dict = dict()
new_project_dict['info'] = final_info_list[1]
out_base = './database/text_classification/'
user_info_path = out_base + 'final-info.json'
with open(user_info_path, 'w') as f:
    f.write(json.dumps(new_project_dict, indent=1, ensure_ascii=False))

new_project_dict = dict()
new_project_dict['info'] = final_info_list[0]
out_base = './database/style_transfer/'
user_info_path = out_base + 'final-info.json'
with open(user_info_path, 'w') as f:
    f.write(json.dumps(new_project_dict, indent=1, ensure_ascii=False))
