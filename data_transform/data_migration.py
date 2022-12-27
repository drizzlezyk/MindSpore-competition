import time
import json


with open("./groups_members_info.json", 'r', encoding='UTF-8') as load_f:
    patri_list = json.load(load_f)

with open("competition_20221111/competition/groups.json", 'r', encoding='UTF-8') as load_f:
    group_list = json.load(load_f)

with open("competition_20221111/competition/score_new.json", 'r', encoding='UTF-8') as load_f:
    score_list = json.load(load_f)

with open("competition_20221111/competition/group_score_new.json", 'r', encoding='UTF-8') as load_f:
    group_score = json.load(load_f)

with open("competition_20221111/competition/project.json", 'r', encoding='UTF-8') as load_f:
    project_list = json.load(load_f)


def print_info():
    competition_count = [0] * 3
    for parti in patri_list:
        group_id = parti['group_id']
        for group in group_list:
            if group['id'] == group_id:
                competition_id = int(group['relate_competition_id'])
                competition_count[competition_id-1] += 1
                break
    print(competition_count)


# ------- 用户信息 ----------
user_info_list = [[] * 1 for _ in range(3)]
final_user_info_list = [[] * 1 for _ in range(3)]
identity_list = ['student', 'teacher', 'developer', '']


def user_process():
    for parti in patri_list:
        group_id = parti['group_id']
        temp_dict = dict()
        for group in group_list:
            if group['id'] == group_id:

                temp_dict["name"] = parti['name']
                temp_dict["city"] = parti['loc_city']
                temp_dict["email"] = parti['email']
                temp_dict["phone"] = parti['phone']
                temp_dict["account"] = parti['username']
                temp_dict["identity"] = identity_list[int(parti['identity_type'])-1]
                temp_dict["province"] = parti['loc_province']
                detail = {
                    "detail1": parti['detail1'],
                    "detail2": parti['detail2']
                }
                temp_dict["detail"] = detail

                if group['is_individual'] == '0':
                    temp_dict["tid"] = parti['group_id']
                    temp_dict["role"] = "leader" if group['leader_id'] == parti['user_id'] else ""
                else:
                    temp_dict["role"] = ""
                    temp_dict["tid"] = ""
                competition_id = int(group['relate_competition_id']) - 1

                user_info_list[competition_id].append(temp_dict)

                if group['period'] == '2':
                    final_user_info_list[competition_id].append(temp_dict)

                break


user_process()
print(len(user_info_list[0]))
print(len(user_info_list[1]))

print(len(final_user_info_list[0]))
print(len(final_user_info_list[1]))
print(len(final_user_info_list[2]))


# ------- 提交信息 ----------
submit_dict = dict()
submit_list = [[] * 1 for _ in range(3)]
submit_list_final = [[] * 1 for _ in range(3)]


def submission_process():
    for score in score_list:
        score_id = score['id']
        temp_dict = dict()
        for g_score in group_score:
            if score_id == g_score['score_id']:
                group_id = g_score['group_id']
                for group in group_list:
                    if group['id'] == group_id:
                        for parti in patri_list:
                            if parti['user_id'] == group['leader_id']:
                                temp_dict["id"] = score_id
                                temp_dict["tid"] = group_id if group['is_individual'] == '0' else ''
                                temp_dict["account"] = parti['username'] \
                                    if group['is_individual'] == '1' else ''
                                temp_dict["path"] = identity_list[int(parti['identity_type'])-1]

                                temp_date = time.strptime(score['create_time'][:19],
                                                          "%Y-%m-%d %H:%M:%S")
                                # 转换成时间戳
                                timestamp = time.mktime(temp_date)
                                temp_dict["submit_at"] = int(timestamp)

                                score_str = score['score']
                                if score_str and score['status_info'] == '评分成功' \
                                        or score['status_info'] == '评估成功' and float(score_str) > 0:
                                    temp_dict["score"] = float(score_str)
                                    temp_dict["status"] = "success"
                                else:
                                    temp_dict["score"] = 0.0
                                    temp_dict["status"] = "failed"
                                temp_dict["date"] = score['create_time'][:10]

                                competition_id = int(group['relate_competition_id']) - 1

                                submit_time = time.mktime(time.strptime(score['create_time'][:10],
                                                                        '%Y-%m-%d'))
                                final_time = time.mktime(time.strptime('2022-11-1', '%Y-%m-%d'))

                                diff = int(submit_time) - int(final_time)
                                if group['period'] == '2' and diff >= 0:
                                    submit_list_final[competition_id].append(temp_dict)
                                elif diff < 0:
                                    submit_list[competition_id].append(temp_dict)
                                break


submission_process()
print(submit_list_final[0])
print("submit pre: ",
      len(submit_list[0]),
      len(submit_list[1]),
      len(submit_list[2]))

print("submit final: ",
      len(submit_list_final[0]),
      len(submit_list_final[1]),
      len(submit_list_final[2]))

# ----- group -------

new_group_list = [[] * 1 for _ in range(3)]
final_new_group_list = [[] * 1 for _ in range(3)]


def group_process():
    for group in group_list:
        if group['is_individual'] == '0' :
            temp_dict = dict()
            temp_dict['id'] = group['id']
            temp_dict['name'] = group['name']
            competition_id = int(group['relate_competition_id']) - 1

            new_group_list[competition_id].append(temp_dict)

            if group['period'] == '2':
                final_new_group_list[competition_id].append(temp_dict)


print("team pre: ", len(new_group_list[0]), len(new_group_list[1]))
print("team final: ", len(final_new_group_list[0]), len(final_new_group_list[1]))
group_process()

# --------- repo ------------
new_project_list = [[] * 1 for _ in range(3)]
final_new_project_list = [[] * 1 for _ in range(3)]
count = [0]


def repos_process():
    for group in group_list:
        for parti in patri_list:
            if group['leader_id'] == parti['user_id']:
                temp_dict = dict()
                temp_dict['tid'] = group['id'] if \
                    group['is_individual'] == '0' else ''
                temp_dict['account'] = parti['username'] if \
                    group['is_individual'] == '1' else ''

                for project in project_list:
                    if group['relate_project_id'] == project['id']:
                        temp_dict['owner'] = parti['username']
                        temp_dict['repo'] = project['name']
                        competition_id = int(group['relate_competition_id']) - 1

                        new_project_list[competition_id].append(temp_dict)
                        count[0] += 1
                        if group['period'] == '2':
                            final_new_project_list[competition_id].append(temp_dict)
                        break


repos_process()
print('count ', count)
print("project pre: ", len(new_project_list[0]), len(new_project_list[1]))
print("project final: ", len(final_new_project_list[0]), len(final_new_project_list[1]))


# ------------------------- output all -----------------------
comp_list = ['style_transfer', 'text_classification', 'image_classification']


def write_output(stage, comp_id):
    total_dict = dict()
    total_dict['repos'] = new_project_list[comp_id]
    total_dict['submissions'] = submit_list[comp_id]
    total_dict['competitors'] = user_info_list[comp_id]
    total_dict['teams'] = new_group_list[comp_id]

    path = './database_new/' + comp_list[comp_id] + '/' + stage + '.json'
    with open(path, 'w') as file:
        file.write(json.dumps(total_dict, indent=1, ensure_ascii=False))


write_output('pre', comp_id=0)
write_output('final', comp_id=0)

write_output('pre', comp_id=1)
write_output('final', comp_id=1)

write_output('pre', comp_id=2)
write_output('final', comp_id=2)
