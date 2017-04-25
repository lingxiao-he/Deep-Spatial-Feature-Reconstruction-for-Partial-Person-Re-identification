function Dictionary = multiscale(test)
B=reshape(test,size(test,1)*size(test,2),size(test,3));
%B = [];
for ii = 1:size(test, 1)-1
    for jj = 1:size(test, 2)-1
        C = test(ii:ii+1,jj:jj+1,:);
        temp = max(max(C));
        B = [B;temp(:)'];
    end
end
for ii = 1:size(test, 1)-2
    for jj = 1:size(test, 2)-2
        C = test(ii:ii+2,jj:jj+2,:);
        temp = max(max(C));
        B = [B;temp(:)'];
    end
end
for ii = 1:size(test, 1)-3
    for jj = 1:size(test, 2)-3
        C = test(ii:ii+3,jj:jj+3,:);
        temp = max(max(C));
        B = [B;temp(:)'];
    end
end


temp = max(max(C));;
B = [B;temp(:)'];



for ii = 1:size(B,1)
    Dictionary(ii,:) = B(ii,:)/norm(B(ii,:),2);
end
for ii = 1:size(Dictionary,1)
    for jj = 1:size(Dictionary,2)
        if (isnan(Dictionary(ii,jj)))
            Dictionary(ii,jj)=0;
        end
    end
end
Dictionary = double(Dictionary);