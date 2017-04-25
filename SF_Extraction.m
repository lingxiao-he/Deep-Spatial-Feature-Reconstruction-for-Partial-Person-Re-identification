function output = SF_Extraction(input)
output=reshape(input,size(input,1)*size(input,2),size(input,3));
for i = 1:size(output,1)
    output(i,:) = output(i,:)/norm(output(i,:),2);
end
for i = 1:size(output,1)
    for j = 1:size(output,2)
        if (isnan(output(i,j)))
            output(i,j)=0;
        end
    end
end
output = double(output);