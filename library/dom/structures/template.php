<?php

/**
 * Template wraps a html file in an object. The string cannot be changed, 
 * unless it has containers that can be replaced by objects.
 * In order to add an object to the container, you need to respect
 * the allowed format for containers in your html file, which is: 
 * <replace location="random name"></replace>
 * Once added, the element object will overwrite the container.
 * Elements added to this object will be treated as children nodes of the block
 * element that wraps the string.
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\text;
use library\dom\elements\components\elementFactory;
use library\dom\object;
use app\model\file;


class template extends components{
    
         
    public function __construct($file,$wrapper = "div") {
        $path = TEMPLATE_PATH.DS.$file.".php"; 
        $string = file::read($path);
        if($string){
            $this->root = elementFactory::createByTag($wrapper);
        } else {
            throw new \Exception("Could not create template from file.");
        }
        $pattern = '#(<replace location="\w+"><\/replace>)#';
        $this->components = preg_split($pattern,$string,-1,PREG_SPLIT_DELIM_CAPTURE);    
    }
 
    public function addComponent(object $component,$location) {
        $flag = false;
        $id = 'location="'.$location.'"';
        foreach($this->components as $key => $child){
            if(strpos($child,$id) !== false){
                $this->components[$key] = $component;
                $flag = true;
                break;
            }
        }
        if(!$flag){
            throw new \Exception("Could not add element to the template: $component");
        }
        
    }
    public final function save(){
        foreach($this->components as $child){
            if($child instanceof object){
                $this->root->addComponent($child);
            } else {
                $this->root->addComponent(new text($child));
            }
            
        }
        return $this->root;    
        
    }



    

    
}
