function accuracy = calrank(distance,rankvalue,info)
for k = 1:length(rankvalue)
    count = 0;
    for i = 1:size(distance,1)
        rankinfo = distance(:,i);
        %[score,position] = sort(rankinfo,'ascend');
        [score,position] = sort(rankinfo,info);
        for j = 1:rankvalue(k);
            if (position(j)==i)
                count = count + 1;
                break;
            end
        end
        accuracy(k) = count/size(distance,1);
    end
end