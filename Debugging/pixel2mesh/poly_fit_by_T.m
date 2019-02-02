function [h_select,out] = poly_fit_by_T(im,h_samples,lane,gt,h_select)
out = [];
show = 0;
order = 2;

reduce = 1;
color = {'r','g','b','m','y','k'};
[h,w,~] = size(im);
T = [gt(1:3);0,gt(4:5);0,gt(6),1];
if reduce
    T = T .* [w, h, 1;
        w, h, 1;
        w, h, 1;];
    T(1,:) = T(1,:)/w;
    T(2,:) = T(2,:)/h;
end
T_ = inv(T);

vp(1) = T_(1,2)/T_(3,2);
vp(2) = T_(2,2)/T_(3,2);

if show
    figure(1),subplot(211),imshow(im)
    for j=1:size(lane,1)
        idx = find(lane(j,:)~=-2);
        hold on,plot(lane(j,idx),h_samples(idx),'-*y','linewidth',2);
    end
    hold on,plot([0,w],[vp(2) vp(2)]*h,':y')
end
for n=1:size(lane,1)
    x1 = lane(n,:)';
    y10 = h_samples';
    eff_idx = find((x1~=-2) & (y10>vp(2)*h*1.05));
    x1 = x1(eff_idx);
    y1 = y10(eff_idx);
    if reduce
        x1 = x1/w;
        y1 = y1/h;
        y10 = y10/h;
    end
    
    x1_ = T(1,1)*x1 + T(1,2)*y1 + T(1,3);
    y1_ = T(2,1)*x1 + T(2,2)*y1 + T(2,3);
    z1_ = T(3,1)*x1 + T(3,2)*y1 + T(3,3);
    x1_ = x1_./z1_;
    y1_ = y1_./z1_;
    
    y10_ = T(2,2)*y10 + T(2,3);
    z10_ = T(3,2)*y10 + T(3,3);
    y10_ = y10_./z10_;

    Y = ones(length(y1_),order+1);
    for j=1:order
        Y(:,order-j+1) = y1_.^j;
    end
    weight = inv(Y'*Y)*(Y'*x1_);
    x1__ = Y*weight;
    
    Y0 = ones(length(y10_),order+1);
    for j=1:order
        Y0(:,order-j+1) = y10_.^j;
    end
    x10__ = Y0*weight;
 
    x1_s = T_(1,1)*x1__ + T_(1,2)*y1_ + T_(1,3);
    y1_s = T_(2,1)*x1__ + T_(2,2)*y1_ + T_(2,3);
    z1_s = T_(3,1)*x1__ + T_(3,2)*y1_ + T_(3,3);
    x1_s = x1_s./z1_s;
    y1_s = y1_s./z1_s;
    
    x10_s = T_(1,1)*x10__ + T_(1,2)*y10_ + T_(1,3);
    z10_s = T_(3,2)*y10_ + T_(3,3);
    x10_s = x10_s./z10_s;
    
    [c2,ia2,ib2] = intersect(h_samples,h_select);
    [~,loc] = sort(ib2,'ascend');
    out = x10_s(ia2(loc)); 
    idx = find(h_select/h<vp(2)*1.05);
    if ~isempty(idx)
        out(idx) = out(idx(end)+1);
        h_select(idx) = h_select(idx(end)+1);
    end
    h_select = h_select /h;
    
    if show
        subplot(212),hold on,plot(x1_,y1_,'+','linewidth',2,'color',color{n})
        subplot(212),hold on,plot(x1__,y1_,'*','linewidth',2,'color',color{n})
        subplot(211),hold on,plot(x1_s*w,y1_s*h,'-*','linewidth',2,'color',color{n})
        subplot(211),hold on,plot(out(n,:)*w,h_select,'ob','linewidth',2)
    end
end