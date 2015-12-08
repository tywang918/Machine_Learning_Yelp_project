Last_userID = 0;
Last_user_group = 1;

result = zeros(9, 40);

review_flag = zeros(40, 1);

for i = 1 : 85043
    i
    if (not(strcmp(review_user(i, 1), Last_userID)))
        result(Last_user_group, :) = result(Last_user_group, :) + review_flag';
        review_flag = zeros(40, 1);
        Last_userID = review_user(i, 1);
        for j = 1 : 21461
            if (strcmp(review_user(i, 1), user(j, 1)))
                Last_user_group = user_group(j, 1) + 1;
            end
        end
        review_flag(review_url_group(i, 1) + 1, 1) = 1;
    else
        review_flag(review_url_group(i, 1) + 1, 1) = 1;
    end
end
result(Last_user_group, :) = result(Last_user_group, :) + review_flag';