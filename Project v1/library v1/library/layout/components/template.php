<?php

/**
 * Template wraps a html file in a string. The string cannot be changed, 
 * unless it has containers that can be replaced by element objects.
 * In order to add an element object to the container, you need to respect
 * the allowed format for containers in your html file, which is: 
 * <div location="id of the element to be added"></div>
 * Once added, the element object will overwrite the container.
 * Elements added to this object will be treated as children nodes of the block
 * element that wraps the string.
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;
use library\layout\elements\element;
use app\model\file;


class template extends component{
    
    public function __construct($file,$tag = "div") {
        $path = VIEW_PATH.DS.$file.".php"; 
        $string = file::read($path);
        if($string){
            $this->tag = $tag;
            $this->closeTag = true;
            $this->value = $string;
        } else {
            throw new \Exception("Could not create template from file.");
        }
    }
    public function addElement(element $element) {
        $id = 'location="'.$element->getId().'"';
        $value = $this->getValue();
        if(strpos($value,$id) !== false){
            $this->elements[] = $element;
            $this->addChild($element);
        } else {
            throw new \Exception("Could not add element to component: $element");
        }
        
    }
    public final function renderTemplate(){
        foreach($this->elements as $element){
            $id = 'location="'.$element->getId().'"';
            $pattern = "#<div $id><\/div>#";
            $this->setValue(preg_replace($pattern, $element, $this->value));
            $element->setRenderFlag(true);
            
        }

        
    }


    

    
}
