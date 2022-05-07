M = 27  # length of english alphabet + 1
A_ORD = 97  # ASCII code of letter 'a'


def cnt_substrings(n, m, s, t):
    cnt = 0
    max_w_len = min(n, m) + 1

    cards_cnt = counting(t)

    for w_len in range(1, max_w_len):
        cnt += check_substrings(n, cards_cnt, s, w_len)

    return cnt


def check_substrings(n, cards_cnt, s, w_len):
    sub_cnt = 0

    for i in range(n - w_len + 1):
        part_cnt = counting(s[i:(i + w_len)])

        res = all(part_cnt[j] <= cards_cnt[j] for j in range(M))
        if res:
            # if len(s[i:(i + w_len)])>1:
            #     print('ok',s[i:(i + w_len)],part_cnt)
            sub_cnt += 1

    return sub_cnt


def counting(arr):
    cnt_arr = [0] * M

    for i in range(len(arr)):
        cnt_arr[ord(arr[i]) - A_ORD] += 1

    return cnt_arr


info = input().split()
n = int(info[0])
m = int(info[1])
s = [x for x in input()]
t = [x for x in input()]

# for letter in t:
#     if letter not in s:
#         t.remove(letter)
# m = len(t)

print(cnt_substrings(n, m, s, t))
