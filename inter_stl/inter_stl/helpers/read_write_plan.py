import numpy as np
import csv
import os

def plan_to_csv(rvars,hvars,ids,other_names,robot_name,scenario_name='',path=''):
    assert (len(rvars) == len(hvars))
    # write rvars (list of numpy array) and hvars
    # as well as the ids (indiciating pre-post-impact etc.)
    if robot_name[0] == '/':
        fn = f"plan_{scenario_name}_{robot_name[1::]}.csv"
    else:
        fn = f"plan_{scenario_name}_{robot_name}.csv"

    # than read the existing csv file
    with open(os.path.join(path,fn),'w',newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write rvars
        writer.writerow(['rvars'])
        for rvar in rvars:
            writer.writerows(rvar)
            writer.writerow([])
        # write hvars
        writer.writerow(['hvars'])
        for hvar in hvars:
            writer.writerows(hvar)
            writer.writerow([])
        # write ids
        writer.writerow(['ids'])
        for id in ids:
            writer.writerow([id])
            writer.writerow([])
        # write other_names
        writer.writerow(['other_names'])
        for other_name in other_names:
            writer.writerow([other_name])
            writer.writerow([])

    return 1

def csv_to_plan(robot_name,scenario_name='',path=''):
    rvars, hvars, ids, other_names = [], [], [], []
    # if first character in robot_name = '/' (it's a namespace) we remove it
    if robot_name[0] == '/':
        fn = f"plan_{scenario_name}_{robot_name[1::]}.csv"
    else:
        fn = f"plan_{scenario_name}_{robot_name}.csv"

    # than read the existing csv file
    with open(os.path.join(path,fn),'r') as csvfile:
        reader = csv.reader(csvfile)
        section = None
        temp_array = []
        for row in reader:
            if not row:
                if section == 'rvars' and temp_array:
                    rvars.append(np.array(temp_array))
                elif section == 'hvars' and temp_array:
                    hvars.append(np.array(temp_array))
                elif section == 'ids' and temp_array:
                    ids.append(temp_array[0])
                elif section == 'other_names' and temp_array:
                    other_names.append(temp_array[0])
                temp_array = []
            elif row[0] in ['rvars','hvars','ids','other_names']:
                section = row[0]
            else:
                if section == 'rvars' or section == 'hvars':
                    temp_array.append([float(x) for x in row])
                elif section == 'ids':
                    temp_array.append(str(row[0]))
                elif section == 'other_names':
                    other_names.append(str(row[0]))
    return rvars, hvars, ids, other_names


def zonotopes_to_csv(X0s,Xfs,ids,other_names,robot_name,scenario_name='',path=''):
    assert (len(X0s) == len(Xfs))
    # write rvars (list of numpy array) and hvars
    # as well as the ids (indiciating pre-post-impact etc.)
    if robot_name[0] == '/':
        fn = f"zonotopes_{scenario_name}_{robot_name[1::]}.csv"
    else:
        fn = f"zonotopes_{scenario_name}_{robot_name}.csv"

    # than read the existing csv file
    with open(os.path.join(path,fn),'w',newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write X0s
        writer.writerow(['X0s'])
        for X0 in X0s:
            print("X0.x: ",X0.x)
            writer.writerows(X0.x)
            writer.writerows(X0.Gdiag)
            writer.writerow([])
        # write Xfs
        writer.writerow(['Xfs'])
        for Xf in Xfs:
            writer.writerows(Xf.x)
            writer.writerows(Xf.Gdiag)
            writer.writerow([])
        # write ids
        writer.writerow(['ids'])
        for id in ids:
            writer.writerow([id])
            writer.writerow([])
        # write other_names
        writer.writerow(['other_names'])
        for other_name in other_names:
            writer.writerow([other_name])
            writer.writerow([])

    return 1    
