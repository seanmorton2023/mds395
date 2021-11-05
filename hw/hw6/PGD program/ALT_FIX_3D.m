function [ Model, e, eB ] = ALT_FIX_3D( X,ec )
    nmax=1000; % Maximal mode number
    vmax=2400; % Maximal iterations for each mode
    ebr=1e-6;  % Convergence criterion of each mode
    
    K=size(X,3); 

    f=zeros(size(X));
    bta1=0;
    bta2=0;
    bta3=0;
    B.F1=ones(1,size(X,1))';
    B.F2=ones(1,size(X,2))';
    B.F3=ones(1,size(X,3))';
    fna=0;
       
    for k=1:K
        norm_X(k)=norm(X(:,:,k));
    end

    %---- Altenating fix point algorithm --------
    for n=1:nmax
        R=X-fna;  
        for k=1:K
            norm_R(k)=norm(R(:,:,k));          
        end
        e(n)=norm(norm_R./norm_X);
        if e(n)<ec
            break;
        end
        M=0;
        for k=1:K 
            D=diag(B.F3(k,:));
            M=M+R(:,:,k)*B.F2*D;
        end
        B.F1=M/((B.F2'*B.F2)*(B.F3'*B.F3)) ;

        M=0;
        for k=1:K 
            D=diag(B.F3(k,:));
            M=M+R(:,:,k)'*B.F1*D;      
        end
        B.F2=M/((B.F1'*B.F1)*(B.F3'*B.F3)) ;

        for k=1:K
            B.F3(k,:)=((B.F2'*B.F2)*(B.F1'*B.F1))\diag(B.F1'*R(:,:,k)*B.F2);
        end         
            br1=norm(B.F1);
            br2=norm(B.F2);
            br3=norm(B.F3);

        for v=1:vmax     
                   M=0;
                    for k=1:K
                        D=diag(B.F3(k,:));
                        M=M+R(:,:,k)*B.F2*D;
                    end
                    B.F1=M/((B.F2'*B.F2)*(B.F3'*B.F3)) ;

                    M=0;
                    for k=1:K  
                        D=diag(B.F3(k,:));
                        M=M+R(:,:,k)'*B.F1*D;      
                    end
                    B.F2=M/((B.F1'*B.F1)*(B.F3'*B.F3)) ;

                    for k=1:K
                        B.F3(k,:)=((B.F2'*B.F2)*(B.F1'*B.F1))\diag(B.F1'*R(:,:,k)*B.F2);
                    end 
                       bt1=norm(B.F1);
                       bt2=norm(B.F2);
                       bt3=norm(B.F3);       
                       eB.e1(v,n)=norm(bt1-bta1)/br1;
                       eB.e2(v,n)=norm(bt2-bta2)/br2;
                       eB.e3(v,n)=norm(bt3-bta3)/br3;

               if eB.e1(v,n)<ebr&&eB.e2(v,n)<ebr&&eB.e3(v,n)<ebr

                   for k=1:K
                   D=diag(B.F3(k,:));    
                   f(:,:,k)=B.F1*D*B.F2';  
                   end

                   Model.F1(:,n)=B.F1;
                   Model.F2(:,n)=B.F2;
                   Model.F3(:,n)=B.F3;
                   fna=fna+f;
                   bta1=0;
                   bta2=0;
                   bta3=0;
                   B.F2=ones(1,size(X,2))';
                   B.F3=ones(1,size(X,3))';
                   break     
               else 
                   bta1=bt1;
                   bta2=bt2;
                   bta3=bt3;
               end

        end
              
    end

end

