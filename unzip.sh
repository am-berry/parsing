#!/bin/bash                                               
                                                          
# nocache 7z e *.bz2                                      
# nocache 7z e *.xz                                       
                                                          
for f in * ; do                                           
    if [[ $f == *.zst ]]; then                            
        cat res.csv | wc -l                               
        nocache unzstd -d --rm "$f"                       
        for g in * ; do                                   
            if [[ ! -d $g ]]; then                        
                case $g in *.*) continue;; esac           
                nocache mv  -- "$g" "./src/data/${g}.json"
                cargo run                                 
                rm "./src/data/${g}.json"                 
            fi                                            
        done                                              
    fi                                                    
done                                                      
                                                  
