review_url_group = zeros(85043, 1);

for i = 1 : 85043
    i
    for j = 1 : 25887
        if (strcmp(review_url(i, 1), url(j, 1)))
            review_url_group(i, 1) = url_group(j, 1);
            break;
        end
    end
end