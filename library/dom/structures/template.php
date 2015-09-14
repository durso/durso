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
use library\dom\dom;
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
        $elements = dom::buildTree($string, $offsetTag);
        if(!is_null($this->root)){
            foreach($elements as $element){
                $this->root->addComponent($element);
            }
        } else {
            $this->collection = $elements;
        }
    }

    public final function save(){    
        return is_null($this->root) ? $this->collection : $this->root;
    }



    

    
}
