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
use library\dom\elements\components\elementFactory;
use library\dom\elements\element;
use library\dom\elements\paired;
use app\model\file;


class template extends components{
    private $collection;
         
    public function __construct($wrapper = false) {
        if($wrapper){
            $this->root = elementFactory::createByTag($wrapper);
        }    
    }
 
    public function create($file,$offsetTag = false){
        $string = file::read($file);
        if($string === false){
            throw new \Exception("Could not open file");
        }
        $pattern = '#(<[^!>]*[^\/][/]*>)#';
        $components = preg_split($pattern,$string,-1,PREG_SPLIT_DELIM_CAPTURE);
        $list = array();
        $offset = false;
        foreach($components as $key => $value){
            $value = trim($value);
            $len = strlen($value);
            if(!$len){
                continue;
            }
            
            if($value[0] == "<" && $value[1] != "!"){
                if($value[1] != "/"){
                    $pos = strpos($value, " ");
                    if($pos){
                        $tag = substr($value,1,$pos - 1);
                    } else {
                        $tag = substr($value,1,-1);
                    }
                    if($tag == 'html'){
                        continue;
                    }
                    if($offsetTag && !$offset){
                        if($offsetTag == $tag){
                            $offset = true;
                        } else {
                            continue;
                        }
                    }
                    $element = elementFactory::createByTag($tag);
                } else {
                    array_pop($list);
                    continue;
                }
            } else {
                $element = elementFactory::createText($value);
            }
            if(empty($list)){      
                if(is_null($this->root)){
                    $this->collection[] = $element;
                } else {
                    $this->root->addComponent($element);
                }
            } else {
                $parent = end($list);
                reset($list);
                $parent->addComponent($element);
            }
            if($element instanceof paired){
                $list[] = $element;
            }
            if($element instanceof element){
                if($pos){
                    $attr = substr($value,$pos,-1);
                    $element->stringToAttr($attr);
                }
            }
        }
    }

    public final function save(){
        if(is_null($this->root)){     
            return $this->collection;
        }
        return $this->root;    
    }



    

    
}
